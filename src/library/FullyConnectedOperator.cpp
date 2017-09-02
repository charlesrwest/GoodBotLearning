#include "FullyConnectedOperator.hpp"
#include "UtilityFunctions.hpp"

using namespace GoodBot;

FullyConnectedOperator::FullyConnectedOperator(std::string name, const std::string& inputBlobName, const std::string& outputBlobName, int64_t inputSize, int64_t outputSize, int64_t batchSize, const std::string& weightFillType, const std::string& biasFillType) : InputBlobName(inputBlobName), OutputBlobName(outputBlobName), InputSize(inputSize), OutputSize(outputSize), BatchSize(batchSize), WeightFillType(weightFillType), BiasFillType(biasFillType)
{
SetName(name);
}

std::vector<std::string> FullyConnectedOperator::GetInputBlobNames() const
{
return {InputBlobName};
}

std::vector<std::string> FullyConnectedOperator::GetOutputBlobNames() const
{
return {OutputBlobName};
}

std::vector<std::string> FullyConnectedOperator::GetTrainableBlobNames() const
{
return {MakeWeightBlobName(Name()), MakeBiasBlobName(Name())};
}

std::vector<std::vector<int64_t>> FullyConnectedOperator::GetTrainableBlobShapes() const
{
return {{OutputSize, InputSize}, {OutputSize}};
}

std::vector<caffe2::OperatorDef> FullyConnectedOperator::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> result;

result.emplace_back();
caffe2::OperatorDef& fullyConnectedOperator = result.back();
fullyConnectedOperator.set_type("FC");
fullyConnectedOperator.add_input(InputBlobName);
fullyConnectedOperator.add_input(MakeWeightBlobName(Name()));
fullyConnectedOperator.add_input(MakeBiasBlobName(Name()));
fullyConnectedOperator.add_output(OutputBlobName);

return result;
}

std::vector<caffe2::OperatorDef> FullyConnectedOperator::GetNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> result;

//Setup weights/biases
result.emplace_back();
caffe2::OperatorDef& weightOperator = result.back();
weightOperator.set_type(WeightFillType);
caffe2::Argument& weightShape = *weightOperator.add_arg();
weightShape.set_name("shape");
weightShape.add_ints(OutputSize); //Number of nodes in this layer
weightShape.add_ints(InputSize); //Number of inputs to this layer
weightOperator.add_output(MakeWeightBlobName(Name()));

result.emplace_back();
caffe2::OperatorDef& biasOperator = result.back();
biasOperator.set_type(BiasFillType);
biasOperator.add_output(MakeBiasBlobName(Name()));
caffe2::Argument& biasShape = *biasOperator.add_arg();
biasShape.set_name("shape");
biasShape.add_ints(OutputSize); //Number of nodes in this layer

return result; 
}
