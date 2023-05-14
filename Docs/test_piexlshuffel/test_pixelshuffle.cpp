#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <vector>
#include <chrono>
#include "utils.h"
#include "PixelShufflePlugin.h"

// data and time


// stuff we know about the network and the input/output blobs
static const int Batch = 1;
static const int INPUT_C = 4;
static const int INPUT_H = 10;
static const int INPUT_W = 10;
static const int Scale = 2;
static const int OUTPUT_W = INPUT_W * Scale;
static const int OUTPUT_H = INPUT_H * Scale;
static const int OUTPUT_C = INPUT_C / (Scale * Scale);

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "upsample";

using namespace nvinfer1;

static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createNetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_C, INPUT_H, INPUT_W});
    //ITensor 为tensor在网络中表示的类  
    assert(data);

    IPluginCreator* pixelshuffle_creator = getPluginRegistry()->getPluginCreator("pixelshuffle_Plugin", "1");
    // 插件的name，getPluginName函数的返回值
    //const PluginFieldCollection* pluginData = pixelshuffle_creator->getFieldNames();

    std::vector<nvinfer1::PluginField> f;
    
    f.emplace_back("scale", &Scale, nvinfer1::PluginFieldType::kINT32, 1);
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();

    //无参数的 create方法
    //IPluginV2 *pixelshuffle_plugin = pixelshuffle_creator->createPlugin("AddScalaPlugin", pluginData,);

    //这个name可以随便设置 不影响。
    IPluginV2 *pixelshuffle_plugin = pixelshuffle_creator->createPlugin("pixelshuffle", &fc);
    IPluginV2Layer* pixelshuffle_layer = network->addPluginV2(&data, 1, *pixelshuffle_plugin);
    //pixelshuffle_layer->setName("pixelshuffle_layer");
    ITensor* resu = pixelshuffle_layer->getOutput(0);
    resu->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*resu);


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

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createNetEngine(maxBatchSize, builder, config, DataType::kFLOAT);
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
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_W * OUTPUT_H * OUTPUT_W * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //input传给 buffers[inputIndex]  送入device上去 

    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_W * OUTPUT_H * OUTPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
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

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("pixelshuffle.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("pixelshuffle.engine", std::ios::binary);
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


    // Subtract mean from image
    float data[INPUT_C * INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++)
        data[i] = i*1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
  
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    // runtime 生成状态的进行时 用来反序列化引擎
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    float output_data[OUTPUT_SIZE];
    // 概率
    
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, output_data, Batch);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nInput:\n\n";
    for(int i =0;i<INPUT_H * INPUT_W;i++)
    {
        std::cout<< data[i]<<" ";
    }
    std::cout<< std::endl;
    std::cout << "\nOutput:\n\n";
    for(int i =0;i<OUTPUT_SIZE;i++)
    {
        std::cout<< output_data[i]<<" ";
    }
    std::cout<< std::endl;
    
    

    return 0;
}
