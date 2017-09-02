#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct MaxPoolModuleParameters
{
std::string Name;
std::string InputBlobName;
std::string OutputBlobName;
int64_t Stride;
int64_t Padding;
int64_t KernelSize;
std::string ImageOrder;
};

class MaxPoolModule : public ComputeModuleDefinition
{
public:
MaxPoolModule(const MaxPoolModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string InputBlobName;
std::string OutputBlobName;
int64_t Stride;
int64_t Padding;
int64_t KernelSize;
std::string ImageOrder;
};

}
