#include "CastModule.hpp"

using namespace GoodBot;

CastModule::CastModule(const CastModuleParameters& inputParameters) : InputBlobName(inputParameters.InputBlobName), TargetDataType(inputParameters.TargetDataType), OutputBlobName(inputParameters.OutputBlobName)
{
SetName(inputParameters.name);
}

std::vector<std::string> CastModule::GetInputBlobNames() const
{
return {InputBlobName};
}

std::vector<std::string> CastModule::GetOutputBlobNames() const
{
return {OutputBlobName};
}

std::vector<caffe2::OperatorDef> CastModule::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("Cast");
operatorDef.add_input(InputBlobName);
operatorDef.add_output(OutputBlobName);

caffe2::Argument& value = *operatorDef.add_arg();
value.set_name("to");
value.set_s("FLOAT");
//value.set_i(TargetDataType);

caffe2::Argument& from_type = *operatorDef.add_arg();
from_type.set_name("from_type");
from_type.set_s("UINT8");

return results;
}
