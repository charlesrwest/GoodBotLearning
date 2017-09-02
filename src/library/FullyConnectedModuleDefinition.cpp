#include "FullyConnectedModuleDefinition.hpp"
#include "FullyConnectedLayerDefinition.hpp"

#include<iostream>

using namespace GoodBot;

FullyConnectedModuleDefinition::FullyConnectedModuleDefinition(const std::string& inputBlobName, const std::vector<int64_t>& numberOfNodesInLayers, const std::string& moduleName, int64_t numberOfInputs, const std::string& weightFillType, const std::string& biasFillType, const std::string& activationType) : moduleName(moduleName)
{
//Construct layers for this module
FullyConnectedLayerDefinitionParameters layerParameters;
layerParameters.numberOfInputs = numberOfInputs;
layerParameters.weightFillType = weightFillType;
layerParameters.biasFillType = biasFillType;
layerParameters.activationType = activationType;

for(int64_t layerIndex = 0; layerIndex < numberOfNodesInLayers.size(); layerIndex++)
{
int64_t numberOfNodesInLayer = numberOfNodesInLayers[layerIndex];

if(layerIndex == 0)
{
layerParameters.inputBlobName = inputBlobName;
}
else
{
layerParameters.inputBlobName = modules[layerIndex - 1]->GetOutputBlobNames()[0];
layerParameters.numberOfInputs = numberOfNodesInLayers[layerIndex-1];
}

layerParameters.numberOfNodes = numberOfNodesInLayer;
layerParameters.layerName = Name() + "_layer" + std::to_string(layerIndex);

AddModule(*(new FullyConnectedLayerDefinition(layerParameters)));
}

//In case of solver, etc modules being added
storedInputBlobNames = modules.front()->GetInputBlobNames();
storedOutputBlobNames = modules.back()->GetOutputBlobNames();
}

std::string FullyConnectedModuleDefinition::Name() const
{
return moduleName;
}

std::vector<std::string> FullyConnectedModuleDefinition::GetInputBlobNames() const
{
return storedInputBlobNames;
}

std::vector<std::string> FullyConnectedModuleDefinition::GetOutputBlobNames() const
{
return storedOutputBlobNames;
}
