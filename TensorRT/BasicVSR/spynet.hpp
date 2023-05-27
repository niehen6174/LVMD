#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include "NvInfer.h"
using namespace nvinfer1;

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
ITensor* SpyNet(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* ref, ITensor* supp){
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

    return flow_0;
}

