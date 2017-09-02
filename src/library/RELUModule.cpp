#include "RELUModule.hpp"

using namespace GoodBot;

RELUModule::RELUModule(const RELUModuleParameters& inputParameters) : InputBlobName(inputParameters.InputBlobName), OutputBlobName(inputParameters.OutputBlobName)
{
SetName(inputParameters.name);
}

std::vector<std::string> RELUModule::GetInputBlobNames() const
{
return {InputBlobName};
}

std::vector<std::string> RELUModule::GetOutputBlobNames() const
{
return {OutputBlobName};
}

std::vector<caffe2::OperatorDef> RELUModule::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("Relu");
operatorDef.add_input(InputBlobName);
operatorDef.add_output(OutputBlobName);

return results;
}
