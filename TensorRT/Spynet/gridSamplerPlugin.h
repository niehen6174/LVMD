/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef GRID_SAMPLER_PLUGIN_H
#define GRID_SAMPLER_PLUGIN_H

#include "gridSampler.h"
#include "NvInferPlugin.h"
#include "common/plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2DynamicExt and BaseCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

using torch::detail::GridSamplerInterpolation;
using torch::detail::GridSamplerPadding;
using torch::detail::GridSamplerDataType;

class GridSamplerPlugin : public IPluginV2DynamicExt
{
public:
    GridSamplerPlugin(const std::string name, GridSamplerInterpolation interpolationMode, GridSamplerPadding paddingMode
        , bool alignCorners);
    GridSamplerPlugin(const std::string name, int inputChannel, int inputHeight,
        int inputWidth, int gridHeight, int gridWidth, GridSamplerInterpolation interpolationMode,
        GridSamplerPadding paddingMode, bool alignCorners, DataType type);
    GridSamplerPlugin(const std::string name, const void* serial_buf, size_t serial_size);
    // It doesn't make sense to make GridSamplerPlugin without arguments, so we delete default constructor.
    GridSamplerPlugin() = delete;
    ~GridSamplerPlugin() override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;
    void detachFromContext() override;

    // IPluginV2 Methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    size_t mBatch;
    size_t mInputWidth, mInputHeight, mInputChannel, mGridHeight, mGridWidth;
    GridSamplerInterpolation mInterpolationMode;
    GridSamplerPadding mPaddingMode;
    bool mAlignCorners;
    std::string mNamespace;
    DataType mType;


protected:
    // For deprecated methods, To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
};

class GridSamplerPluginCreator : public BaseCreator
{
public:
    GridSamplerPluginCreator();

    ~GridSamplerPluginCreator() override;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator);
} // namespace plugin

} // namespace nvinfer1

#endif // GRID_SAMPLER_PLUGIN_H
