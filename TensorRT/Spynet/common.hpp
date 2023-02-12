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

ITensor* BasicModule(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, int block_idx){
    //conv 1
    IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, 32, DimsHW{ 7, 7 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.0.conv.bias"]);
    conv_1->setStrideNd(DimsHW{ 1, 1 });
    conv_1->setPaddingNd(DimsHW{ 3, 3 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kRELU );

    ITensor* x1 = leaky_relu_1->getOutput(0);
    //conv 2
    IConvolutionLayer* conv_2 = network->addConvolutionNd(*x1, 64, DimsHW{ 7, 7 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.1.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.1.conv.bias"]);
    conv_2->setStrideNd(DimsHW{ 1, 1 });
    conv_2->setPaddingNd(DimsHW{ 3, 3 });
    IActivationLayer* leaky_relu_2 = network->addActivation(*conv_2->getOutput(0), ActivationType::kRELU);

    ITensor* x2 = leaky_relu_2->getOutput(0);
    //conv 3
    IConvolutionLayer* conv_3 = network->addConvolutionNd(*x2, 32, DimsHW{ 7, 7 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.2.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.2.conv.bias"]);
    conv_3->setStrideNd(DimsHW{ 1, 1 });
    conv_3->setPaddingNd(DimsHW{ 3, 3 });
    IActivationLayer* leaky_relu_3 = network->addActivation(*conv_3->getOutput(0), ActivationType::kRELU );

    ITensor* x3 = leaky_relu_3->getOutput(0);
    //conv 4
    IConvolutionLayer* conv_4 = network->addConvolutionNd(*x3, 16, DimsHW{ 7, 7 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.3.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.3.conv.bias"]);
    conv_4->setStrideNd(DimsHW{ 1, 1 });
    conv_4->setPaddingNd(DimsHW{ 3, 3 });
    IActivationLayer* leaky_relu_4 = network->addActivation(*conv_4->getOutput(0), ActivationType::kRELU );

    ITensor* x4 = leaky_relu_4->getOutput(0);
    //conv 5
    IConvolutionLayer* conv_5 = network->addConvolutionNd(*x4, 2, DimsHW{ 7, 7 }, weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.4.conv.weight"], 
                                                            weightMap["basic_module." + std::to_string(block_idx) + ".basic_module.4.conv.bias"]);
    conv_5->setStrideNd(DimsHW{ 1, 1 });
    conv_5->setPaddingNd(DimsHW{ 3, 3 });
    
    ITensor* x5 = conv_5->getOutput(0);
    return x5;

}