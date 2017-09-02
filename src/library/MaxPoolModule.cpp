#include "MaxPoolModule.hpp"

using namespace GoodBot;

MaxPoolModule::MaxPoolModule(const MaxPoolModuleParameters& inputParameters) : InputBlobName(inputParameters.InputBlobName), OutputBlobName(inputParameters.OutputBlobName), Stride(inputParameters.Stride), Padding(inputParameters.Padding), KernelSize(inputParameters.KernelSize), ImageOrder(inputParameters.ImageOrder)
{
SetName(inputParameters.Name);
}

std::vector<std::string> MaxPoolModule::GetInputBlobNames() const
{
return {InputBlobName};
}

std::vector<std::string> MaxPoolModule::GetOutputBlobNames() const
{
return {OutputBlobName};
}

std::vector<caffe2::OperatorDef> MaxPoolModule::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("MaxPool");
operatorDef.add_input(InputBlobName);
operatorDef.add_output(OutputBlobName);

caffe2::Argument& stride = *operatorDef.add_arg();
stride.set_name("stride");
stride.set_i(Stride);

caffe2::Argument& pad = *operatorDef.add_arg();
pad.set_name("pad");
pad.set_i(Padding);

caffe2::Argument& kernel = *operatorDef.add_arg();
kernel.set_name("kernel");
kernel.set_i(KernelSize);

caffe2::Argument& order = *operatorDef.add_arg();
order.set_name("order");
order.set_s(ImageOrder);

return results;
}
