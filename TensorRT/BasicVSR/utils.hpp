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

ITensor* ResidualBlocksWithInputConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int num_blocks, int mid_channels, string block_type){
    IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, mid_channels, DimsHW{ 3, 3 }, weightMap["generator." + block_type + "main.2." + std::to_string(block_idx) + ".conv1.weight"], 
                                                            weightMap["generator." + block_type + "main.2." + std::to_string(block_idx) + ".conv1.bias"]);
    conv_1->setStrideNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kLEAKY_RELU);

    ITensor* x1 = leaky_relu_1->getOutput(0);
    for(int i = 0; i < num_blocks; i++){
        x1 = ResidualBlockNoBN(network, weightMap, x1, i, mid_channels, block_type);
    }
    return x1;
}

ITensor* ResidualBlockNoBN(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int block_idx, int mid_channels, string block_type){
    //conv 1
    IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, mid_channels, DimsHW{ 3, 3 }, weightMap["generator." + block_type + "main.2." + std::to_string(block_idx) + ".conv1.weight"], 
                                                            weightMap["generator." + block_type + "main.2." + std::to_string(block_idx) + ".conv1.bias"]);
    conv_1->setStrideNd(DimsHW{ 1, 1 });
    conv_1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kRELU );

    ITensor* x1 = leaky_relu_1->getOutput(0);
    //conv 2
    IConvolutionLayer* conv_2 = network->addConvolutionNd(*x1, mid_channels, DimsHW{ 3, 3 }, weightMap["generator." + block_type + "main.2." + std::to_string(block_idx) + ".conv2.weight"], 
                                                            weightMap["generator." + block_type + "main.2." + std::to_string(block_idx) + ".conv2.bias"]);
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

ITensor* UpsampleBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int Scale){
    IConvolutionLayer* fusion = network->addConvolutionNd(*x, 2 * mid_channels, DimsHW{ 1, 1 }, weightMap["generator.fusion.weight"], 
                                                            weightMap["generator.fusion.bias"]);
    fusion->setStrideNd(DimsHW{ 1, 1 });
    fusion->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*fusion->getOutput(0), ActivationType::kLEAKY_RELU );
    // PixelShufflePack1
    IConvolutionLayer* upsample_conv1 = network->addConvolutionNd(*leaky_relu_1->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["generator.upsample1.upsample_conv.weight"], 
                                                            weightMap["generator.upsample1.upsample_conv.bias"]); 
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
    IActivationLayer* leaky_relu_2 = network->addActivation(*pixelshuffle_layer1->getOutput(0), ActivationType::kLEAKY_RELU );

    // PixelShufflePack2
    IConvolutionLayer* upsample_conv2 = network->addConvolutionNd(*leaky_relu_2->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["generator.upsample2.upsample_conv.weight"], 
                                                            weightMap["generator.upsample2.upsample_conv.bias"]); 
    upsample_conv2->setStrideNd(DimsHW{ 1, 1 });
    upsample_conv2->setPaddingNd(DimsHW{ 1, 1 });
    
    IPluginV2Layer* pixelshuffle_layer2 = network->addPluginV2(*upsample_conv2->getOutput(0), 1, *pixelshuffle_plugin);
    IActivationLayer* leaky_relu_3 = network->addActivation(*pixelshuffle_layer2->getOutput(0), ActivationType::kLEAKY_RELU );

    // conv_hr
    IConvolutionLayer* conv_hr = network->addConvolutionNd(*leaky_relu_3->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["generator.conv_hr.weight"], 
                                                            weightMap["generator.conv_hr.bias"]);
    IActivationLayer* leaky_relu_4 = network->addActivation(*conv_hr->getOutput(0), ActivationType::kLEAKY_RELU );
    // conv_last
    IConvolutionLayer* conv_last = network->addConvolutionNd(*leaky_relu_4->getOutput(0), mid_channels, DimsHW{ 3, 3 }, weightMap["generator.conv_last.weight"], 
                                                            weightMap["generator.conv_last.bais"]);
    // IResizeLayer* img_upsample = network->addResize(*conv_last->getOutput(0));
    // img_upsample->setResizeMode(ResizeMode::kBILINEAR);
    // const float sclaes[] = { 1, 1, 2, 2 };
    // img_upsample ->setScales(sclaes, 4);
    return conv_last->getOutput(0);
}




