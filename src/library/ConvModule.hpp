#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct ConvModuleParameters
{
std::string Name;
std::string InputBlobName;
std::string OutputBlobName;
int64_t InputDepth;
int64_t OutputDepth;
int64_t Stride;
int64_t PaddingSize;
int64_t KernelSize;
};

class ConvModule : public ComputeModuleDefinition
{
public:
ConvModule(const ConvModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<std::string> GetTrainableBlobNames() const override;

virtual std::vector<std::vector<int64_t>> GetTrainableBlobShapes() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

protected:
std::string InputBlobName;
std::string OutputBlobName;
int64_t InputDepth;
int64_t OutputDepth;
int64_t Stride;
int64_t PaddingSize;
int64_t KernelSize;
};

}
