#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common/logging.h"
#include <fstream>
#include <map>
#include <vector>
#include <chrono>
#include <algorithm>
#include "common/cuda_utils.h"
#include "gridSamplerPlugin.h"
#include "utils.hpp"
#include "spyNet.hpp"
// data and time


// stuff we know about the network and the input/output blobs
static const int INPUT_T = 2;
static const int INPUT_C = 1;
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int SCALES = 4;
static const int OUTPUT_SIZE = 2 * INPUT_H * INPUT_W;
static const int MIND_CHANNEL = 64;
//static const int OUTPUT_SIZE = 3;
const char* INPUT_BLOB_NAME = "inputs";
const char* OUTPUT_BLOB_NAME = "outputs";

using namespace nvinfer1;

static Logger gLogger;


// Creat the engine using only the API and not any parser.
ICudaEngine* createNetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string& vsr_wts_name,
                             std::string& spy_wts_name, int mid_channels)
{
    INetworkDefinition* network = builder->createNetworkV2(1);
    std::map<std::string, Weights> weightMap = loadWeights(vsr_wts_name);
    // Create input tensor of shape { 1, INPUT_C, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* lrs = network->addInput(INPUT_BLOB_NAME, dt, Dims4{T, INPUT_C, INPUT_H, INPUT_W});
    assert(supp);

    // lrs forward
    ITensor* lrs_1 = network->addSlice(*lrs, Dims4{1, 0, 0, 0}, Dims4{T-1, INPUT_C, INPUT_H, INPUT_W})->getOutput(0);
    // lrs backward
    ITensor* lrs_2 = network->addSlice(*lrs, Dims4{T-1, 0, 0, 0}, Dims4{T, INPUT_C, INPUT_H, INPUT_W},{-1, 1, 1, 1})->getOutput(0);

    lrs_2 = network->addSlice(*lrs_2, Dims4{1, 0, 0, 0}, Dims4{T-1, INPUT_C, INPUT_H, INPUT_W})->getOutput(0);
    
    std::map<std::string, Weights> spynet_weightMap = loadWeights(spy_wts_name);
    ITensor* flows_forward = SpyNet(network, spynet_weightMap, lrs_1, lrs_2);
    ITensor* flows_backward = SpyNet(network, spynet_weightMap, lrs_2, lrs_1);

    std::vector<ITensor*> outputs;
    float constan_value = 0.0f;
    IConstantLayer* constant_zero = network->addConstant(Dims3{mid_channels, INPUT_H, INPUT_W}, Weights{DataType::kFLOAT, &constan_value, INPUT_H*INPUT_W*mid_channels} );
    ITensor* feat_prop = constant_zero->getOutput(0);

    //flow_warp operation
    IPluginCreator* flowWarp_creator = getPluginRegistry()->getPluginCreator("GridSampler", "1");
    std::vector<nvinfer1::PluginField> f;
    int interpolation_mode = 0; //{Bilinear, Nearest}
    int padding_mode = 0; //{Zeros, Border, Reflection}
    int align_corners = 1; //{false, true}
    f.emplace_back("interpolation_mode", &interpolation_mode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("padding_mode", &padding_mode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("align_corners", &align_corners, nvinfer1::PluginFieldType::kINT32, 1);
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    std::vector<ITensor*> input_data = {supp_list[0], flow_up_init};
    IPluginV2 *flowWarp_plugin = flowWarp_creator->createPlugin("flowWarp", &fc);

    // generate downsampled frames 
    ITensor* lrs_2_temp = network->addSlice(*lrs_2, Dims4{0, 0, 0, 0}, Dims4{1, INPUT_C, INPUT_H, INPUT_W})->getOutput(0);
    feat_prop = network->addConcatenation({feat_prop, lrs_2_temp}, dim = 1)->getOutput(0);
    feat_prop = ResidualBlocksWithInputConv(network, weightMap, feat_prop, mid_channels, 3, 1, "backward_resblocks");
    ITensor* flow;
    for(int i = 1; i < T-1; ++i)
    {
        flow = network->addSlice(*flows_backward, Dims4{T-1-i, 0, 0, 0}, Dims4{1, 2, INPUT_H, INPUT_W})->getOutput(0);
        std::vector<ITensor*> input_data = {feat_prop, flow};
        IPluginV2Layer* flowWarp_layer = network->addPluginV2(input_data.data(), 2, *flowWarp_plugin);
        flow_warp = flowWarp_layer->getOutput(0);
        lrs_2_temp = network->addSlice(*lrs_2, Dims4{i, 0, 0, 0}, Dims4{1, INPUT_C, INPUT_H, INPUT_W})->getOutput(0);
        feat_prop = network->addConcatenation({feat_prop, lrs_2_temp}, dim = 1)->getOutput(0);
        outputs.push_back(feat_prop);
    }
    // vector 翻转
    std::reverse(outputs.begin(),outputs.end());
    // generate upsampled frames
    ITensor* flow, lrs_temp, out_temp, cat_prop;
    IResizeLayer* img_upsample_layer;
    for(int i = 0; i < T; ++i){
        lrs_temp = network->addSlice(*lrs, Dims4{i, 0, 0, 0}, Dims4{1, INPUT_C, INPUT_H, INPUT_W})->getOutput(0);
        if(i > 0){
            flow = network->addSlice(*flows_forward, Dims4{i-1, 0, 0, 0}, Dims4{1, 2, INPUT_H, INPUT_W})->getOutput(0);
            input_data.clear();
            input_data = {feat_prop, flow};
            feat_prop = network->addPluginV2(input_data.data(), 2, *flowWarp_plugin)->getOutput(0);
        }
        feat_prop = network->addConcatenation({feat_prop, lrs_temp}, dim = 1)->getOutput(0);
        feat_prop = ResidualBlocksWithInputConv(network, weightMap, feat_prop, mid_channels, 3, 1, "forward_resblocks");
        
        cat_prop = network->addConcatenation({outputs[i], feat_prop}, dim = 1)->getOutput(0);

        img_upsample_layer = network->addResize(*lrs_temp->getOutput(0));
        img_upsample_layer->setResizeMode(ResizeMode::kBILINEAR);
        const float sclaes[] = { 1, 1, 2, 2 };
        img_upsample_layer ->setScales(sclaes, 4);
        out_temp = UpsampleBlock(network, weightMap, cat_prop, mid_channels, 3, 1, "upsample_block");
        // out_temp += img_upsample_layer->getOutput(0);
        out_temp = network->addElementWise(out_temp, img_upsample_layer->getOutput(0), ElementWiseOperation::kSUM)->getOutput(0);
        outputs[i] = out_temp;
    }
    IConcatenationLayer* concatLayer = network->addConcatenation(outputs, T);
    concatLayer->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*concatLayer->getOutput(0));

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

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string& vsr_wts_name, std::string& spy_wts_name, int mind_channel)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createNetEngine(maxBatchSize, builder, config, DataType::kFLOAT, vsr_wts_name, spy_wts_name, mind_channel);
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
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    std::cout << engine.getNbBindings() << std::endl;
    std::cout << inputIndex << " " << inputIndex2 <<" " << outputIndex<< std::endl;
    assert(engine.getNbBindings() == 3);
    void* buffers[3];
  
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], T * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], T * INPUT_C * INPUT_H * INPUT_W * SCALES * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input,  T * INPUT_C * INPUT_H * INPUT_W  * sizeof(float), cudaMemcpyHostToDevice, stream));
    //input传给 buffers[inputIndex]  送入device上去 

    context.enqueue(T, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], T * INPUT_C * INPUT_H * INPUT_W * SCALES * sizeof(float), cudaMemcpyDeviceToHost, stream));
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
        std::cerr << "./basicvsr -s   // serialize model to plan file" << std::endl;
        std::cerr << "./basicvsr -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    assert(INPUT_H >= 32 && INPUT_H%32 == 0);

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::string vsr_wts_name = "./basicvsr.wts";
    std::string spy_wts_name = "./spynet.wts";
    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream, vsr_wts_name, spy_wts_name, MIND_CHANNEL);
        assert(modelStream != nullptr);
        std::ofstream p("basicvsrplugin.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("basicvsrplugin.engine", std::ios::binary);
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
    float data[T * INPUT_C * INPUT_H * INPUT_W];
    for (int i = 0; i < T * INPUT_C * INPUT_H * INPUT_W; i++)
    {    data[i] = 0.5f;
    }
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
  
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    // runtime 生成状态的进行时 用来反序列化引擎
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    float output_data[OUTPUT_SIZE];
    
    
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, output_data, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nInput:\n\n";
    for(int i =0; i < T * INPUT_C * INPUT_H * INPUT_W; i++)
    {
        std::cout<< data[i]<<" ";
    }
    std::cout<< std::endl;
    std::cout << "\nOutput:\n\n";
    for(int i =0; i < T * SCALES * INPUT_C * INPUT_H * INPUT_W; i++)
    {
        std::cout<< output_data[i]<<" ";
    }
    std::cout<< std::endl;

    return 0;
}
