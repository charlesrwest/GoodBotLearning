#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct SoftMaxLayerDefinitionParameters
{
std::string inputBlobName;
std::string layerName;
};

/**
This class is a straight forward implementation of the "Softmax" operator (depending on mode).  See ComputeModuleDefinition for the function meanings.
*/
class SoftMaxLayerDefinition : public ComputeModuleDefinition
{
public:
SoftMaxLayerDefinition(const SoftMaxLayerDefinitionParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string inputBlobName;
};




























}
