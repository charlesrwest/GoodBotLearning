#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

/**
This class is a straight forward implementation of the "Softmax" operator (depending on mode).  See ComputeModuleDefinition for the function meanings.
*/
class LabelCrossEntropyOperator : public ComputeModuleDefinition
{
public:
LabelCrossEntropyOperator(const std::string& name, const std::string& softmaxBlobName, const std::string& labelBlobName);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string SoftmaxBlobName;
std::string LabelBlobName;
};




























}
