#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct RELUModuleParameters
{
std::string name;
std::string InputBlobName;
std::string OutputBlobName;
};

class RELUModule : public ComputeModuleDefinition
{
public:
RELUModule(const RELUModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string InputBlobName;
std::string OutputBlobName;
};

}
