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
// data and time


// stuff we know about the network and the input/output blobs
static const int INPUT_C = 1;
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int OUTPUT_SIZE = 2 * INPUT_H * INPUT_W;
//static const int OUTPUT_SIZE = 3;
const char* INPUT_BLOB_NAME = "ref";
const char* INPUT_BLOB_NAME2 = "supp";
const char* OUTPUT_BLOB_NAME = "flow";

using namespace nvinfer1;

static Logger gLogger;


// Creat the engine using only the API and not any parser.
ICudaEngine* createNetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string& wts_name)
{
    INetworkDefinition* network = builder->createNetworkV2(1);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    // Create input tensor of shape { 1, INPUT_C, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* ref = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, INPUT_C, INPUT_H, INPUT_W});
    ITensor* supp = network->addInput(INPUT_BLOB_NAME2, dt, Dims4{1, INPUT_C, INPUT_H, INPUT_W});
    //ITensor 为tensor在网络中表示的类  
    assert(ref);
    assert(supp);

    // 标准化 
    const float mean[3] = {0.485, 0.456, 0.406}; // rgb
    const float std[3] = {0.229, 0.224, 0.225};
    Weights Mean{ DataType::kFLOAT, mean, 3 };
    Weights Std{ DataType::kFLOAT, std, 3 };
    IConstantLayer* m = network->addConstant(Dims4{ 1, 3, 1, 1 }, Mean);
    IConstantLayer* s = network->addConstant(Dims4{ 1, 3, 1, 1 }, Std);
    IElementWiseLayer* sub_mean_ref = network->addElementWise(*ref, *m->getOutput(0), ElementWiseOperation::kSUB);
    IElementWiseLayer* std_mean_ref = network->addElementWise(*sub_mean_ref->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);

    IElementWiseLayer* sub_mean_supp = network->addElementWise(*supp, *m->getOutput(0), ElementWiseOperation::kSUB);
    IElementWiseLayer* std_mean_supp = network->addElementWise(*sub_mean_supp->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);
    ITensor* pre_ref = std_mean_ref->getOutput(0);
    ITensor* pre_supp = std_mean_supp->getOutput(0);

 

    std::vector<ITensor*> ref_list{pre_ref};
    std::vector<ITensor*> supp_list{pre_supp};
    
    // generate downsampled frames 
    for(int i=0; i<5; i++)
    {
        IPoolingLayer* pool_1_1 =  network->addPoolingNd(*ref_list[i], nvinfer1::PoolingType::kAVERAGE, DimsHW{2,2});
        pool_1_1->setStrideNd(DimsHW{2,2});
        ref_list.emplace_back(pool_1_1->getOutput(0));
        IPoolingLayer* pool_2_1 =  network->addPoolingNd(*supp_list[i], nvinfer1::PoolingType::kAVERAGE, DimsHW{2,2});
        pool_2_1->setStrideNd(DimsHW{2,2});
        supp_list.emplace_back(pool_2_1->getOutput(0));
    }
    

    
    // list 翻转
    std::reverse(ref_list.begin(),ref_list.end());
    std::reverse(supp_list.begin(),supp_list.end());
    //flow computation
    float constan_value = 0.0f;
    IConstantLayer* constant_zero = network->addConstant(Dims4{1, 2, INPUT_H/32, INPUT_W/32}, Weights{DataType::kFLOAT, &constan_value, INPUT_H*INPUT_W/512} );
    ITensor* flow_up_init = constant_zero->getOutput(0);

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
    IPluginV2Layer* flowWarp_layer = network->addPluginV2(input_data.data(), 2, *flowWarp_plugin);

    ITensor* flowWarp_out = flowWarp_layer->getOutput(0);
    // cat operation
    IConcatenationLayer* concat_flow1_layer = network->addConcatenation(std::vector<ITensor*>{ref_list[0], flowWarp_out, flow_up_init}.data(), 3);
    //send it to convModule
    ITensor* BasicModule1 = BasicModule(network, weightMap, concat_flow1_layer->getOutput(0), 0);
    // BasicModule()  Input tensor with shape (b, 8, h, w).
    //            8 channels contain:[reference image (3), neighbor image (3), initial flow (2)].
    // BasicModule1 Refined flow with shape (b, 2, h, w)

    //# add the residue to the upsampled flow
    IElementWiseLayer* flow_add_0 = network->addElementWise(*flow_up_init, *BasicModule1, ElementWiseOperation::kSUM);
    ITensor* flow_0 = flow_add_0->getOutput(0);
    
    for(int i=1; i<6; i++)
    {
        IResizeLayer* flow_up_layer = network->addResize(*flow_0);
        const float sclaes[] = { 1, 1, 2, 2 };
        flow_up_layer->setScales(sclaes, 4);
        flow_up_layer->setResizeMode(ResizeMode::kLINEAR);
        ITensor* flow_up_out = flow_up_layer->getOutput(0);
        IElementWiseLayer* double_flow_layer = network->addElementWise(*flow_up_out, *flow_up_out, ElementWiseOperation::kSUM);
        flow_up_out = double_flow_layer->getOutput(0);
        std::vector<ITensor*> flowWarp_data = {supp_list[i], flow_up_out};
        IPluginV2Layer* flowWarp_layer_2 = network->addPluginV2(flowWarp_data.data(), 2, *flowWarp_plugin);
        ITensor* flowWarp_out_2 = flowWarp_layer_2->getOutput(0);
        IConcatenationLayer* concat_flow1_layer_2 = network->addConcatenation(std::vector<ITensor*>{ref_list[i], flowWarp_out_2, flow_up_out}.data(), 3);
        ITensor* concat_ouput = concat_flow1_layer_2->getOutput(0);
        ITensor* module_ouput = BasicModule(network, weightMap, concat_ouput, i);
    
        //# add the residue to the upsampled flow
        IElementWiseLayer* flow_add_resi = network->addElementWise(*flow_up_out, *module_ouput, ElementWiseOperation::kSUM);
        flow_0 = flow_add_resi->getOutput(0);
    }
    flow_0->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*flow_0);
    

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
    ICudaEngine* engine = createNetEngine(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
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
    const int inputIndex2 = engine.getBindingIndex(INPUT_BLOB_NAME2);
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    std::cout << engine.getNbBindings() << std::endl;
    std::cout << inputIndex << " " << inputIndex2 <<" " << outputIndex<< std::endl;
    assert(engine.getNbBindings() == 3);
    void* buffers[3];
    //const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    //const int inputIndex2 = engine.getBindingIndex(INPUT_BLOB_NAME2);
    //const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex2], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input,  batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex2], input2, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
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
        std::cerr << "./basicvsr -s   // serialize model to plan file" << std::endl;
        std::cerr << "./basicvsr -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    assert(INPUT_H >= 32 && INPUT_H%32 == 0);

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::string wts_name = "./basicvsr.wts";
    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream, wts_name);
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
    float data[INPUT_C * INPUT_H * INPUT_W];
    float data2[INPUT_C * INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++)
    {    data[i] = 0.5f;
        data2[i] = 0.6f;
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
    doInference(*context, data, data2, output_data, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nInput:\n\n";
    for(int i =0;i<INPUT_C * INPUT_H * INPUT_W;i++)
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
