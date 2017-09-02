#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct CastModuleParameters
{
std::string name;
std::string InputBlobName;
caffe2::TensorProto::DataType TargetDataType;
std::string OutputBlobName;
};

class CastModule : public ComputeModuleDefinition
{
public:
CastModule(const CastModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string InputBlobName;
caffe2::TensorProto::DataType TargetDataType;
std::string OutputBlobName;
};

}
