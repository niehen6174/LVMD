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

//#include "cookbookHelper.hpp"
#include <iostream>
#include <vector>
#include <map>
#include "NvInfer.h"
#include "macros.h"
#include "utils.h"
#include <assert.h>


namespace nvinfer1
{
class PixelShufflePlugin : public IPluginV2IOExt
{
private:
    const std::string name_;
    std::string       namespace_;
    struct
    {
        int scale;
        int nElement;
        int InputChannel, InputHeight, InputWidth;
        int OutputChannel, OutputHeight, OutputWidth;
    } m_;

public:
    PixelShufflePlugin() = delete;
    PixelShufflePlugin(const std::string &name, int scale);
    PixelShufflePlugin(const std::string &name, const void *buffer, size_t length);
    ~PixelShufflePlugin();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    Dims        getOutputDimensions(int32_t index, Dims const *inputs, int32_t nbInputDims) noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    int32_t     enqueue(int32_t batchSize, void const *const *inputs, void *TRT_CONST_ENQUEUE *outputs, void *workspace, cudaStream_t stream) noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType        getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    bool            isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    bool            canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    void            attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void            detachFromContext() noexcept override;
    IPluginV2IOExt *clone() const noexcept override;
    //Method inherited from IPluginV2IOExt
    void configurePlugin(PluginTensorDesc const *in, int32_t nbInput, PluginTensorDesc const *out, int32_t nbOutput) noexcept override;
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept override;
};

class PixelShufflePluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    PixelShufflePluginCreator();
    ~PixelShufflePluginCreator();
    const char *                 getPluginName() const noexcept override;
    const char *                 getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2 *                  createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
    IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *                 getPluginNamespace() const noexcept override;
};
    REGISTER_TENSORRT_PLUGIN(PixelShufflePluginCreator);
} // namespace nvinfer1