#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include "NvInfer.h"

using namespace nvinfer1;

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
// ITensor* utils_MeanStd(INetworkDefinition* network,ITensor* input){
//     //const float mean[3] = {0.485, 0.456, 0.406}; // rgb
//     //const float std[3] = {0.229, 0.224, 0.225};
//     // const float mean[1] = {0.449}; // rgb
//     // const float std[1] = {0.226};
//     float constant_value = 0.449f;
//     Weights Mean{ DataType::kFLOAT, &constant_value, 1 };
//     IConstantLayer* m = network->addConstant(Dims4{ 1, 1, 1, 1 }, Mean);
//     IElementWiseLayer* sub_mean = network->addElementWise(*input, *m->getOutput(0), ElementWiseOperation::kDIV);
//     return sub_mean->getOutput(0);
//     // Weights Std{ DataType::kFLOAT, std, 3 };
//     // //Std.values = std;
//     // IConstantLayer* s = network->addConstant(Dims4{ 1, 3, 1, 1 }, Std);
//     // IElementWiseLayer* std_mean = network->addElementWise(*sub_mean->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);
//     // return std_mean->getOutput(0);
    
// }

ITensor* ResidualBlockNoBN(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int block_idx, int mid_channels){
    //conv 1
    IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, mid_channels, DimsHW{ 3, 3 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]);
    conv_1->setStrideNd(DimsHW{ 1, 1 });
    conv_1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kRELU );

    ITensor* x1 = leaky_relu_1->getOutput(0);
    //conv 2
    IConvolutionLayer* conv_2 = network->addConvolutionNd(*x1, mid_channels, DimsHW{ 3, 3 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.1.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.1.conv.bias"]);
    conv_2->setStrideNd(DimsHW{ 1, 1 });
    conv_2->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_2 = network->addActivation(*conv_2->getOutput(0), ActivationType::kRELU);
    ITensor* x2 = leaky_relu_2->getOutput(0);
    
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *scval = 0.2;
    Weights scale{ DataType::kFLOAT, scval, 1 };
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *shval = 0.0;
    Weights shift{ DataType::kFLOAT, shval, 1 };
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *pval = 1.0;
    Weights power{ DataType::kFLOAT, pval, 1 };

    IScaleLayer* scaled = network->addScale(*x2, ScaleMode::kUNIFORM, shift, scale, power);
    IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x, ElementWiseOperation::kSUM);
    return ew1->getOutput(0);
}

ITensor* BasicModule(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int block_idx, int mid_channels, int num_blocks){
    //conv 1
    IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, mid_channels, DimsHW{ 1, 1 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]);
    conv_1->setStrideNd(DimsHW{ 1, 1 });
    conv_1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kRELU );

    ITensor* resu_input = leaky_relu_1->getOutput(0);
    ITensor* resu_out;
    for(int i = 0; i < num_blocks; i++){
        resu_out = ResidualBlockNoBN(network, weightMap, resu_input, 10, 64);
    }
    return resu_out;

}

// # upsample
// self.fusion = nn.Conv2d(
//     mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
// self.upsample1 = PixelShufflePack(
//     mid_channels, mid_channels, 2, upsample_kernel=3)
// self.upsample2 = PixelShufflePack(
//     mid_channels, 64, 2, upsample_kernel=3)
// self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
// self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
// self.img_upsample = nn.Upsample(
//     scale_factor=4, mode='bilinear', align_corners=False)
ITensor* UpsampleBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int Scale){
    IConvolutionLayer* fusion = network->addConvolutionNd(*x, 2 * mid_channels, DimsHW{ 1, 1 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]);
    fusion->setStrideNd(DimsHW{ 1, 1 });
    fusion->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*fusion->getOutput(0), ActivationType::kRELU );
    // PixelShufflePack1
    IConvolutionLayer* upsample_conv1 = network->addConvolutionNd(*leaky_relu_1->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]); 
    upsample_conv1->setStrideNd(DimsHW{ 1, 1 });
    upsample_conv1->setPaddingNd(DimsHW{ 1, 1 });
    IPluginCreator* pixelshuffle_creator = getPluginRegistry()->getPluginCreator("pixelshuffle_Plugin", "1");
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back("scale", &Scale, nvinfer1::PluginFieldType::kINT32, 1);
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    IPluginV2 *pixelshuffle_plugin = pixelshuffle_creator->createPlugin("pixelshuffle", &fc);
    IPluginV2Layer* pixelshuffle_layer1 = network->addPluginV2(*upsample_conv1->getOutput(0), 1, *pixelshuffle_plugin);
    IActivationLayer* leaky_relu_2 = network->addActivation(*pixelshuffle_layer1->getOutput(0), ActivationType::kRELU );

    // PixelShufflePack2
    IConvolutionLayer* upsample_conv2 = network->addConvolutionNd(*leaky_relu_2->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]); 
    upsample_conv2->setStrideNd(DimsHW{ 1, 1 });
    upsample_conv2->setPaddingNd(DimsHW{ 1, 1 });
    
    IPluginV2Layer* pixelshuffle_layer2 = network->addPluginV2(*upsample_conv2->getOutput(0), 1, *pixelshuffle_plugin);
    IActivationLayer* leaky_relu_3 = network->addActivation(*pixelshuffle_layer2->getOutput(0), ActivationType::kRELU );

    IConvolutionLayer* upsample_conv2 = network->addConvolutionNd(*leaky_relu_3->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]); 
    
    IConvolutionLayer* upsample_conv2 = network->addConvolutionNd(*pixelshuffle_layer1->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]); 
    upsample_conv2->setStrideNd(DimsHW{ 1, 1 });
    upsample_conv2->setPaddingNd(DimsHW{ 1, 1 });

    IActivationLayer* leaky_relu_4 = network->addActivation(*upsample_conv2->getOutput(0), ActivationType::kRELU );
}




