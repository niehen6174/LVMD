/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PixelShufflePlugin.h"

// 用于计算的 kernel
__global__ void pixelShuffelKernel(const float *input, float *output, const int sacle, const int nElement, 
    const int InputChannel,const int InputHeight,const int InputWidth,const int OutputChannel,const int OutputHeight,const int OutputWidth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int upscale_factor = sacle;
    if (idx >= nElement)
        return;

    int n = idx / (InputChannel * InputHeight * InputWidth);
    int c = (idx / (InputHeight * InputWidth)) % InputChannel;
    int h = (idx / InputWidth) % InputHeight;
    int w = idx % InputWidth;
    
    int oc = c / (upscale_factor * upscale_factor);
    int oh = h * upscale_factor + c % upscale_factor;
    int ow = w * upscale_factor + (c / upscale_factor) % upscale_factor;

    int output_idx = n * OutputChannel * OutputHeight * OutputWidth +
                        oc * OutputHeight * OutputWidth + 
                        oh * OutputWidth + 
                        ow;

    output[output_idx] = input[idx];
}
namespace
{
static const char *PLUGIN_NAME {"pixelshuffle_Plugin"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
// 这里各成员函数按照被调用顺序或重要程度顺序排列
// class PixelShufflePlugin
PixelShufflePlugin::PixelShufflePlugin(const std::string &name, int scale):
    name_(name)
{
    
    m_.scale = scale;
}

PixelShufflePlugin::PixelShufflePlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    
    memcpy(&m_, buffer, sizeof(m_));
}

PixelShufflePlugin::~PixelShufflePlugin()
{
    
    return;
}

IPluginV2IOExt *PixelShufflePlugin::clone() const noexcept
{
    
    auto p = new PixelShufflePlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t PixelShufflePlugin::getNbOutputs() const noexcept
{
    
    return 1;
}

DataType PixelShufflePlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    
    return inputTypes[0];
}

Dims PixelShufflePlugin::getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept
{
    int outputChannel = int(inputs[0].d[0] / (m_.scale * m_.scale));
    int outputHeight = inputs[0].d[1] * m_.scale;
    int outputWidth = inputs[0].d[2] * m_.scale;
    return Dims3(outputChannel, outputHeight, outputWidth);
}

bool PixelShufflePlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept
{
    
    switch (pos)
    {
    case 0:
        return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void PixelShufflePlugin::configurePlugin(PluginTensorDesc const *in, int32_t nbInput, PluginTensorDesc const *out, int32_t nbOutput) noexcept
{
    assert(nbInput == 1);
    assert(nbOutput == 1);
    assert(in[0].dims.nbDims == 3);
    
    m_.scale = 2;
    m_.InputChannel = in[0].dims.d[0];
    m_.InputHeight = in[0].dims.d[1];
    m_.InputWidth = in[0].dims.d[2];
    m_.OutputChannel = int(m_.InputChannel / (m_.scale * m_.scale));
    m_.OutputHeight = m_.InputHeight * m_.scale;
    m_.OutputWidth = m_.InputWidth * m_.scale;

    int nElement = 1;
    for (int i = 0; i < in[0].dims.nbDims; ++i)
        nElement *= in[0].dims.d[i];
    m_.nElement = nElement;
    return;
}

size_t PixelShufflePlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    
    return 0;
}

int32_t PixelShufflePlugin::enqueue(int32_t batchSize, void const *const *inputs, void *TRT_CONST_ENQUEUE *outputs, void *workspace, cudaStream_t stream) noexcept
{
    
    int  nElement = batchSize * m_.nElement;
    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    pixelShuffelKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<float *>(outputs[0]), 
            m_.scale, nElement, m_.InputChannel,m_.InputHeight,m_.InputWidth,m_.OutputChannel,m_.OutputHeight,m_.OutputWidth);
    return 0;
}

void PixelShufflePlugin::destroy() noexcept
{
    
    delete this;
    return;
}

int32_t PixelShufflePlugin::initialize() noexcept
{
    
    return 0;
}

void PixelShufflePlugin::terminate() noexcept
{
    
    return;
}

size_t PixelShufflePlugin::getSerializationSize() const noexcept
{
    
    return sizeof(m_);
}

void PixelShufflePlugin::serialize(void *buffer) const noexcept
{
    
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void PixelShufflePlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    
    namespace_ = pluginNamespace;
    return;
}

const char *PixelShufflePlugin::getPluginNamespace() const noexcept
{
    
    return namespace_.c_str();
}

const char *PixelShufflePlugin::getPluginType() const noexcept
{
    
    return PLUGIN_NAME;
}

const char *PixelShufflePlugin::getPluginVersion() const noexcept
{
    
    return PLUGIN_VERSION;
}

bool PixelShufflePlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    
    return false;
}

bool PixelShufflePlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    
    return true;
}

void PixelShufflePlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    
    return;
}

void PixelShufflePlugin::detachFromContext() noexcept
{
    
    return;
}

// class PixelShufflePluginCreator
PluginFieldCollection    PixelShufflePluginCreator::fc_ {};
std::vector<PluginField> PixelShufflePluginCreator::attr_;

PixelShufflePluginCreator::PixelShufflePluginCreator()
{
    
    attr_.clear();
    attr_.emplace_back(PluginField("scale", nullptr, PluginFieldType::kINT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

PixelShufflePluginCreator::~PixelShufflePluginCreator()
{
    
}

// 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
IPluginV2 *PixelShufflePluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    
    int scale = 0;
    std::map<std::string, int *> parameterMap {{"scale", &scale}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    return new PixelShufflePlugin(name, scale);
}

IPluginV2 *PixelShufflePluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    
    return new PixelShufflePlugin(name, serialData, serialLength);
}

void PixelShufflePluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    
    namespace_ = pluginNamespace;
    return;
}

const char *PixelShufflePluginCreator::getPluginNamespace() const noexcept
{
    
    return namespace_.c_str();
}

const char *PixelShufflePluginCreator::getPluginName() const noexcept
{
    
    return PLUGIN_NAME;
}

const char *PixelShufflePluginCreator::getPluginVersion() const noexcept
{
    
    return PLUGIN_VERSION;
}

const PluginFieldCollection *PixelShufflePluginCreator::getFieldNames() noexcept
{
    
    return &fc_;
}

//REGISTER_TENSORRT_PLUGIN(PixelShufflePluginCreator);

} // namespace nvinfer1
