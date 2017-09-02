#include "ScaleModule.hpp"

using namespace GoodBot;

ScaleModule::ScaleModule(const ScaleModuleParameters& inputParameters) : InputBlobName(inputParameters.InputBlobName), OutputBlobName(inputParameters.OutputBlobName), Scale(inputParameters.Scale)
{
SetName(inputParameters.name);
}

std::vector<std::string> ScaleModule::GetInputBlobNames() const
{
return {InputBlobName};
}

std::vector<std::string> ScaleModule::GetOutputBlobNames() const
{
return {OutputBlobName};
}

std::vector<caffe2::OperatorDef> ScaleModule::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("Scale");
operatorDef.add_input(InputBlobName);
operatorDef.add_output(OutputBlobName);

caffe2::Argument& value = *operatorDef.add_arg();
value.set_name("scale");
value.set_f(Scale);

return results;
}
