#pragma once

#include "CompositeComputeModuleDefinition.hpp"

namespace GoodBot
{

struct VGG16Parameters
{
std::string Name;
std::string InputBlobName;
std::string OutputBlobName;
std::string TrainingExpectedOutputBlobName;
std::string TestExpectedOutputBlobName;
int64_t BatchSize;
};

class VGG16 : public CompositeComputeModuleDefinition
{
public:
VGG16(const VGG16Parameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

protected:
int64_t LastModuleIndex;
};

}
