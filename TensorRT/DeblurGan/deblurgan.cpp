#include "cuda_runtime_api.h"
#include "logging.h"
#include <chrono>
#include "utils.hpp"

#include "plugin/postprocess.h"
#include "plugin/preprocess.h"
// data and time


// stuff we know about the network and the input/output blobs
static const int INPUT_H = 720;
static const int INPUT_W = 1280;
static const int CHANNELS = 3;
static const int INPUT_SIZE = CHANNELS * INPUT_H * INPUT_W;
static const int OUTPUT_SIZE = INPUT_SIZE;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createNetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,std::string& wts_name)
{
    INetworkDefinition* network = builder->createNetworkV2(1);

    
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ CHANNELS, INPUT_H, INPUT_W});
    //ITensor 为tensor在网络中表示的类  
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    // Custom preprocess (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))

    Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W };
    IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");
    IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);
    IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);
    preprocess_layer->setName("preprocess_layer");
    ITensor* prep = preprocess_layer->getOutput(0);
    //network  add ReflectionPad2d
    int pad_size = 3;
    ISliceLayer* reflectpad1 = network->addSlice(prep,(0, 0, -1*pad_size, -1*pad_size), (maxBatchSize , CHANNELS, INPUT_H+2*pad_size, INPUT_W+2*pad_size), (1, 1, 1, 1));
    reflectpad1->setmode(SliceMode::kREFLECT);
    ITensor* reflect_out = reflectpad1->getOutput(0);

    //add conv-IN-relu 输入参数第三个为输入 data第四个为输出channel数 第五个位卷积核大小，第六个为卷积步长，第七个为pad size第八个为 wts中索引值
    ITensor* out_conv1 = convInLeaky(network, weightMap, reflect_out, 64, 7, 1, 0, 1, "");

    ITensor* out_conv2 = convInLeaky(network, weightMap, out_conv1, 128, 3, 2, 1, 4, "");

    ITensor* out_conv3 = convInLeaky(network, weightMap, out_conv2, 256, 3, 2, 1, 7, "");

    // resnetblock 
    ITensor* resout = out_conv3;
    for(int i=0; i< 9, i++)
    {
        resout = ResBlock(network, weightMap, resout, 10+i);
    }
    ITensor* convTran_out = convTranInLeaky(network, weightMap, resout,128, 3, 2 , 1, 19);
    //Todo out_padding
    ITensor* convTran_out2 = convTranInLeaky(network, weightMap, convTran_out,64, 3, 2 , 1, 22);

    //Dims3
    auto const tensor_shape = convTran_out2->getDimensions(); 
    int pad_size = 3;
    ISliceLayer* reflectpad2 = network->addSlice(prep,(0, 0, -1*pad_size, -1*pad_size), (tensor_shape.d[0] , tensor_shape.d[1], tensor_shape.d[2]+2*pad_size, tensor_shape.d[3]+2*pad_size), (1, 1, 1, 1));
    reflectpad2->setmode(SliceMode::kREFLECT);
    ITensor* reflect_out2 = reflectpad2->getOutput(0);
    IConvolutionLayer* conv26 = network->addConvolutionNd(reflect_out2, 3, DimsHW{7, 7}, weightMap["26.weight"], weightMap["26.bias"]);
    auto lr = network->addActivation(conv26->getOutput(0), ActivationType::kTANH);
    ITensor* out = lr->getOutput(0);

    Postprocess postprocess{ maxBatchSize, out->getDimensions().d[0], out->getDimensions().d[1], out->getDimensions().d[2] };
    IPluginCreator* postprocess_creator = getPluginRegistry()->getPluginCreator("postprocess", "1");
    IPluginV2 *postprocess_plugin = postprocess_creator->createPlugin("postprocess_plugin", (PluginFieldCollection*)&postprocess);
    IPluginV2Layer* postprocess_layer = network->addPluginV2(&out, 1, *postprocess_plugin);
    postprocess_layer->setName("postprocess_layer");

    ITensor* final_tensor = postprocess_layer->getOutput(0);
    final_tensor->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*final_tensor);

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 << 20);
    //config用来设置最大工作空间 以及构建engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // Don't need the network any more
    network->destroy();
    // Release host memory
    
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string& wts_name)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createNetEngine(maxBatchSize, builder, config, DataType::kFLOAT,wts_name);
    assert(engine != nullptr);
    // Serialize the engine
    (*modelStream) = engine->serialize();
    // 序列化Serialize the network to a stream.
    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //input传给 buffers[inputIndex]  送入device上去 

    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //buffers[outputIndex] 传给output 传回host
    cudaStreamSynchronize(stream);
    //等待这个异步 stream 执行完毕（TRT 的前向预测执行是异步的，）
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./test_add -s   // serialize model to plan file" << std::endl;
        std::cerr << "./test_add -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s" && argc == 4) {
        IHostMemory* modelStream{nullptr};

        std::string wts_name = argv[2];
        std::string engine_name = argv[3];

        APIToModel(1, &modelStream,wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        std::string engine_name = std::string(argv[2]);
        std::string img_dir = std::string(argv[3]);

        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

    } else {
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

  

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
  
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    // runtime 生成状态的进行时 用来反序列化引擎
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create GPU buffers on device	
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(uint8_t)));

    std::vector<uint8_t> input(BATCH_SIZE * INPUT_H * INPUT_W * INPUT_C);
    std::vector<uint8_t> outputs(BATCH_SIZE * OUTPUT_SIZE);

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    for (int f = 0; f < file_names.size(); f++) {

        for (int b = 0; b < BATCH_SIZE; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f]);
            if (img.empty()) continue;
            memcpy(input.data() + b * INPUT_H * INPUT_W * INPUT_C, img.data, INPUT_H * INPUT_W * INPUT_C);
        }

        CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, outputs.data(), BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    cv::Mat frame = cv::Mat(INPUT_H , INPUT_W , CV_8UC3, outputs.data());
    cv::imwrite("../_" + file_names[0] + ".png", frame);


    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    

    return 0;
}
