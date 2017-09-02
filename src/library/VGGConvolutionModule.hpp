#pragma once

#include "CompositeComputeModuleDefinition.hpp"

namespace GoodBot
{

struct VGGConvolutionModuleParameters
{
std::string Name;
std::string InputBlobName;
std::string OutputBlobName;
int64_t InputDepth;
int64_t OutputDepth;
int64_t NumberOfConvolutions;
};

class VGGConvolutionModule : public CompositeComputeModuleDefinition
{
public:
VGGConvolutionModule(const VGGConvolutionModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

protected:
int64_t LastConvLayerIndex;
};

}
