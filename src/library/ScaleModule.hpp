#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct ScaleModuleParameters
{
std::string name;
std::string InputBlobName;
std::string OutputBlobName;
float Scale;
};

class ScaleModule : public ComputeModuleDefinition
{
public:
ScaleModule(const ScaleModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string InputBlobName;
std::string OutputBlobName;
float Scale;
};

}
