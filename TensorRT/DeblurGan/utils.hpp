#ifdef DEBUG
    #define WHERE_AM_I()                          \
        do                                        \
        {                                         \
            printf("%14p[%s]\n", this, __func__); \
        } while (0);
#else
    #define WHERE_AM_I()
#endif // ifdef DEBUG

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define ALIGN_TO(X, Y)    (CEIL_DIVIDE(X, Y) * (Y))

#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include "NvInfer.h"
#include "plugin/instanceNormalizationPlugin.h"
using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}
static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


ITensor* convInLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int index_, std::string pre_index) {
    //Weights emptywts{DataType::kFLOAT, nullptr, 0};  bias 为0的情况下用
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[pre_index + std::to_string(index_) + ".weight"], weightMap[pre_index + std::to_string(index_) + ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    index_++;

    IPluginCreator* addIN_creator = getPluginRegistry()->getPluginCreator("InstanceNormalization_TRT", "1");
    // 插件的name，getPluginName函数的返回值
    //const PluginFieldCollection* pluginData = addscalar_creator->getFieldNames();
    std::vector<nvinfer1::PluginField> f;
    float epsilon = 1e-5f;
    float *mean = (float*)weightMap[std::to_string(index_) + ".running_mean"].values;
    float *var = (float*)weightMap[std::to_string(index_) + ".running_mean"].values;
    int len = weightMap[std::to_string(index_) + ".running_var"].count;
    f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scales", mean, nvinfer1::PluginFieldType::kFLOAT32, len);
    f.emplace_back("bias", var, nvinfer1::PluginFieldType::kFLOAT32, len);
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    //无参数的 create方法
    //IPluginV2 *addscalar_plugin = addscalar_creator->createPlugin("AddScalaPlugin", pluginData,);
    //这个name可以随便设置 不影响。
    if(pre_index.size == 1)
    std::string pre_plugin = "";
    else
    std::string pre_plugin = "res";
    IPluginV2 *addIN_plugin = addIN_creator->createPlugin("INplugin" + pre_plugin + std::to_string(index_), &fc);
    IPluginV2Layer* addIN_layer = network->addPluginV2(conv1->getOutput(0), 1, *addIN_plugin);
    //addscalar_layer->setName("addscalar_layer");
    ITensor* IN_resu = addIN_layer->getOutput(0);
    // 添加relu激活函数
    auto lr = network->addActivation(IN_resu, ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr->output();
}
ITensor* ResBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int index_)
{
   // 这里没有使用reflectpad 直接在卷积的时候featuremap 填充0
    std::string pre_idx = std::to_string(index_) + "conv_block."
    ITensor* out_conv = convInLeaky(network, weightMap, x, 256, 3, 1, 1, 1, pre_idx);
    IElementWiseLayer* ew = network->addElementWise(*out_conv->getOutput(0), *x, ElementWiseOperation::kSUM);

    return ew->getOutput(0);
}
//ConvTranspose2d -- InstanceNorm2d -- ReLU

ITensor* convTranInLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int index_, std::string pre_index) {
    
    IConvolutionLayer* conv1 = network->addDeconvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[pre_index + std::to_string(index_) + ".weight"], weightMap[pre_index + std::to_string(index_) + ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    ITensor* conv1_output = conv1->getOutput(0);
    auto const shape = conv1_output->getDimensions();
    assert(shape.nbDims == 4);
    ISliceLayer* Deconv_outpad = network->addSlice(prep,(0, 0, 0, 0), (shape.d[0] , shape.d[1], shape.d[2]+1, shape.d[3]+1), (1, 1, 1, 1));
    Deconv_outpad->setmode(SliceMode::kCLAMP);
    index_++;
    

    IPluginCreator* addIN_creator = getPluginRegistry()->getPluginCreator("InstanceNormalization_TRT", "1");
    std::vector<nvinfer1::PluginField> f;
    float epsilon = 1e-5f;
    float *mean = (float*)weightMap[std::to_string(index_) + ".running_mean"].values;
    float *var = (float*)weightMap[std::to_string(index_) + ".running_mean"].values;
    int len = weightMap[std::to_string(index_) + ".running_var"].count;
    f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scales", mean, nvinfer1::PluginFieldType::kFLOAT32, len);
    f.emplace_back("bias", var, nvinfer1::PluginFieldType::kFLOAT32, len);
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
   
    IPluginV2 *addIN_plugin = addIN_creator->createPlugin("INplugin" + std::to_string(index_), &fc);
    IPluginV2Layer* addIN_layer = network->addPluginV2(Deconv_outpad->getOutput(0), 1, *addIN_plugin);
    ITensor* IN_resu = addIN_layer->getOutput(0);
    auto lr = network->addActivation(IN_resu, ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr->output();
}