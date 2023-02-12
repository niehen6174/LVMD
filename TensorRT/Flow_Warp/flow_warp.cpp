#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common/logging.h"
#include <fstream>
#include <map>
#include <vector>
#include <chrono>
#include "common/utils.h"
#include "gridSamplerPlugin.h"

// stuff we know about the network and the input/output blobs
static const int INPUT_C = 1;
static const int INPUT_H = 3;
static const int INPUT_W = 3;
static const int GRID_H = 3;
static const int GRID_W = 3;
static const int OUTPUT_SIZE = INPUT_C * GRID_H * GRID_W;

const char* INPUT_BLOB_NAME = "lr";
const char* INPUT_BLOB_NAME2 = "grid";
const char* OUTPUT_BLOB_NAME = "output";

using namespace nvinfer1;

static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createNetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(1);
    // 0 为静态 1位动态
    // Create input tensor of shape { 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* lr = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, INPUT_C,INPUT_H, INPUT_W});
    ITensor* grid = network->addInput(INPUT_BLOB_NAME2, dt, Dims4{1, 2, GRID_H, GRID_W});
    //ITensor 为tensor在网络中表示的类  
    assert(lr);
    assert(grid);

    IPluginCreator* gridSampler_creator = getPluginRegistry()->getPluginCreator("GridSampler", "1");
    // 插件的name，getPluginName函数的返回值
    //const PluginFieldCollection* pluginData = addscalar_creator->getFieldNames();

    std::vector<nvinfer1::PluginField> f;
    int interpolation_mode = 0;
    int padding_mode = 0;
    int align_corners = 1;
    f.emplace_back("interpolation_mode", &interpolation_mode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("padding_mode", &padding_mode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("align_corners", &align_corners, nvinfer1::PluginFieldType::kINT32, 1);
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();

    //无参数的 create方法
    //IPluginV2 *addscalar_plugin = addscalar_creator->createPlugin("AddScalaPlugin", pluginData,);
    std::vector<ITensor*> input_data = {lr, grid};
    
    //这个name可以随便设置 不影响。
    IPluginV2 *gridSampler_plugin = gridSampler_creator->createPlugin("gridsampler", &fc);
    IPluginV2Layer* gridSampler_layer = network->addPluginV2(input_data.data(), 2, *gridSampler_plugin);
    //addscalar_layer->setName("addscalar_layer");
    ITensor* resu = gridSampler_layer->getOutput(0);
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

void doInference(IExecutionContext& context, float* input, float* input2,float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    assert(engine.getNbBindings() == 3);
    void* buffers[3];
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int inputIndex2 = engine.getBindingIndex(INPUT_BLOB_NAME2);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex2], batchSize * GRID_H * GRID_W * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex2], input2, batchSize * GRID_H * GRID_W * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
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
        std::cerr << "./test_plugin -s   // serialize model to plan file" << std::endl;
        std::cerr << "./test_plugin -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("testplugin.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("testplugin.engine", std::ios::binary);
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

    
    float data[INPUT_C * INPUT_H * INPUT_W];
    float grid[GRID_H * GRID_W * 2] = {0.1,0.4,0.7,0.1,0.4,0.7,0.2,0.5,0.8,0.2,0.5,0.8,0.3,0.6,0.9,0.3,0.6,0.9};
    for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++)
        data[i] = 5.0;

    // for (int i = 0; i < GRID_H * GRID_W * 2; i++)
    //     if(i%2 == 0)
    //     grid[i] = 0.1*(i/2) + 0.1;
    //     else
    //     grid[i] = 0.1*((i-1)/2) + 0.1;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
  
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    // runtime 生成状态的进行时 用来反序列化引擎
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    float output_data[OUTPUT_SIZE];
    //输出
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, grid, output_data, 1);
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
    std::cout << "\nGrid:\n\n";
    for(int i =0;i<GRID_H * GRID_W * 2;i++)
    {
        std::cout<< grid[i]<<" ";
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

