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

#include "AddScalaPlugin.h"

// 用于计算的 kernel
__global__ void addScalarKernel(const float *input, float *output, const float scalar, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;

    float _1      = input[index];
    float _2      = _1 + scalar;
    output[index] = _2;
}
namespace
{
static const char *PLUGIN_NAME {"AddScalar_Plugin"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
// 这里各成员函数按照被调用顺序或重要程度顺序排列
// class AddScalarPlugin
AddScalarPlugin::AddScalarPlugin(const std::string &name, float scalar):
    name_(name)
{
    
    m_.scalar = scalar;
}

AddScalarPlugin::AddScalarPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
{
    
    memcpy(&m_, buffer, sizeof(m_));
}

AddScalarPlugin::~AddScalarPlugin()
{
    
    return;
}

IPluginV2IOExt *AddScalarPlugin::clone() const noexcept
{
    
    auto p = new AddScalarPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t AddScalarPlugin::getNbOutputs() const noexcept
{
    
    return 1;
}

DataType AddScalarPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    
    return inputTypes[0];
}

Dims AddScalarPlugin::getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept
{
    
    return inputs[0];
}

bool AddScalarPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept
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

void AddScalarPlugin::configurePlugin(PluginTensorDesc const *in, int32_t nbInput, PluginTensorDesc const *out, int32_t nbOutput) noexcept
{
    
    int nElement = 1;
    for (int i = 0; i < in[0].dims.nbDims; ++i)
        nElement *= in[0].dims.d[i];
    m_.nElement = nElement;
    return;
}

size_t AddScalarPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    
    return 0;
}

int32_t AddScalarPlugin::enqueue(int32_t batchSize, void const *const *inputs, void *TRT_CONST_ENQUEUE *outputs, void *workspace, cudaStream_t stream) noexcept
{
    
    int  nElement = batchSize * m_.nElement;
    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    addScalarKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<float *>(outputs[0]), m_.scalar, nElement);
    return 0;
}

void AddScalarPlugin::destroy() noexcept
{
    
    delete this;
    return;
}

int32_t AddScalarPlugin::initialize() noexcept
{
    
    return 0;
}

void AddScalarPlugin::terminate() noexcept
{
    
    return;
}

size_t AddScalarPlugin::getSerializationSize() const noexcept
{
    
    return sizeof(m_);
}

void AddScalarPlugin::serialize(void *buffer) const noexcept
{
    
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void AddScalarPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    
    namespace_ = pluginNamespace;
    return;
}

const char *AddScalarPlugin::getPluginNamespace() const noexcept
{
    
    return namespace_.c_str();
}

const char *AddScalarPlugin::getPluginType() const noexcept
{
    
    return PLUGIN_NAME;
}

const char *AddScalarPlugin::getPluginVersion() const noexcept
{
    
    return PLUGIN_VERSION;
}

bool AddScalarPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    
    return false;
}

bool AddScalarPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    
    return true;
}

void AddScalarPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    
    return;
}

void AddScalarPlugin::detachFromContext() noexcept
{
    
    return;
}

// class AddScalarPluginCreator
PluginFieldCollection    AddScalarPluginCreator::fc_ {};
std::vector<PluginField> AddScalarPluginCreator::attr_;

AddScalarPluginCreator::AddScalarPluginCreator()
{
    
    attr_.clear();
    attr_.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

AddScalarPluginCreator::~AddScalarPluginCreator()
{
    
}

// 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
IPluginV2 *AddScalarPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    
    float                          scalar = 0;
    std::map<std::string, float *> parameterMap {{"scalar", &scalar}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    return new AddScalarPlugin(name, scalar);
}

IPluginV2 *AddScalarPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    
    return new AddScalarPlugin(name, serialData, serialLength);
}

void AddScalarPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    
    namespace_ = pluginNamespace;
    return;
}

const char *AddScalarPluginCreator::getPluginNamespace() const noexcept
{
    
    return namespace_.c_str();
}

const char *AddScalarPluginCreator::getPluginName() const noexcept
{
    
    return PLUGIN_NAME;
}

const char *AddScalarPluginCreator::getPluginVersion() const noexcept
{
    
    return PLUGIN_VERSION;
}

const PluginFieldCollection *AddScalarPluginCreator::getFieldNames() noexcept
{
    
    return &fc_;
}

//REGISTER_TENSORRT_PLUGIN(AddScalarPluginCreator);

} // namespace nvinfer1
