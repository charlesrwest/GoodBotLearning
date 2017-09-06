#pragma once

#include "CompositeComputeModuleDefinition.hpp"

namespace GoodBot
{

/**
This class represents one or more fully connected layers sequentially connected to each other.  numberOfNodesInLayers determines how many neurons to have at each layer. 
*/
class FullyConnectedModuleDefinition : public CompositeComputeModuleDefinition
{
public:
FullyConnectedModuleDefinition(const std::string& inputBlobName, const std::vector<int64_t>& numberOfNodesInLayers, const std::string& moduleName, int64_t numberOfInputs = 0, const std::string& weightFillType = "XavierFill", const std::string& biasFillType = "ConstantFill", const std::string& activationType = "Tanh");

virtual std::string Name() const override;

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

protected:
std::string moduleName;

std::vector<std::string> storedInputBlobNames;
std::vector<std::string> storedOutputBlobNames;
};































} 
