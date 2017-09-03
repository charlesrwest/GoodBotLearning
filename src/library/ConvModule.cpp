#include "ConvModule.hpp"
#include "UtilityFunctions.hpp"
#include<iostream>

using namespace GoodBot;

ConvModule::ConvModule(const ConvModuleParameters& inputParameters) : InputBlobName(inputParameters.InputBlobName), OutputBlobName(inputParameters.OutputBlobName), InputDepth(inputParameters.InputDepth), OutputDepth(inputParameters.OutputDepth), Stride(inputParameters.Stride), PaddingSize(inputParameters.PaddingSize), KernelSize(inputParameters.KernelSize)
{
SetName(inputParameters.Name);
}

std::vector<std::string> ConvModule::GetInputBlobNames() const
{
return {InputBlobName};
}

std::vector<std::string> ConvModule::GetOutputBlobNames() const
{
return {OutputBlobName};
}

std::vector<std::string> ConvModule::GetTrainableBlobNames() const
{
return {MakeWeightBlobName(Name()), MakeBiasBlobName(Name())};
}

std::vector<std::vector<int64_t>> ConvModule::GetTrainableBlobShapes() const
{
return {{OutputDepth, InputDepth, KernelSize, KernelSize}, {OutputDepth}};
}

std::vector<caffe2::OperatorDef> ConvModule::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("Conv");

operatorDef.add_input(InputBlobName);
operatorDef.add_input(MakeWeightBlobName(Name()));
operatorDef.add_input(MakeBiasBlobName(Name()));

operatorDef.add_output(OutputBlobName);

caffe2::Argument& stride = *operatorDef.add_arg();
stride.set_name("stride");
stride.set_i(Stride);

caffe2::Argument& pad = *operatorDef.add_arg();
pad.set_name("pad");
pad.set_i(PaddingSize);

caffe2::Argument& kernel = *operatorDef.add_arg();
kernel.set_name("kernel");
kernel.set_i(KernelSize);

return results;
}

std::vector<caffe2::OperatorDef> ConvModule::GetNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& weight_operator = results.back();
weight_operator.set_type("XavierFill");
caffe2::Argument& weight_shape = *weight_operator.add_arg();
weight_shape.set_name("shape");
weight_shape.add_ints(OutputDepth);
weight_shape.add_ints(InputDepth);
weight_shape.add_ints(KernelSize);
weight_shape.add_ints(KernelSize);

weight_operator.add_output(MakeWeightBlobName(Name()));

results.emplace_back();
caffe2::OperatorDef& bias_operator = results.back();
bias_operator.set_type("ConstantFill");
bias_operator.add_output(MakeBiasBlobName(Name()));
caffe2::Argument& bias_shape = *bias_operator.add_arg();
bias_shape.set_name("shape");
bias_shape.add_ints(OutputDepth); //Number of nodes in this layer

return results;
}
