#include "SoftMaxLayerDefinition.hpp"
#include "UtilityFunctions.hpp"
#include<iostream>

using namespace GoodBot;

SoftMaxLayerDefinition::SoftMaxLayerDefinition(const SoftMaxLayerDefinitionParameters& inputParameters) : inputBlobName(inputParameters.inputBlobName)
{
SetName(inputParameters.layerName);
}

std::vector<std::string> SoftMaxLayerDefinition::GetInputBlobNames() const
{
return {inputBlobName};
}

std::vector<std::string> SoftMaxLayerDefinition::GetOutputBlobNames() const
{
return {Name()};
}

std::vector<caffe2::OperatorDef> SoftMaxLayerDefinition::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& softMax = results.back();

softMax.set_type("Softmax");
softMax.add_input(inputBlobName);
softMax.add_output(GetOutputBlobNames()[0]);

return results;
}

