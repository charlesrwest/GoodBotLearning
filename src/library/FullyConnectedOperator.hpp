#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

class FullyConnectedOperator : public ComputeModuleDefinition
{
public:
FullyConnectedOperator(std::string name, const std::string& inputBlobName, const std::string& outputBlobName, int64_t inputSize, int64_t outputSize, int64_t batchSize = 1, const std::string& weightFillType = "XavierFill", const std::string& biasFillType = "ConstantFill");

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<std::string> GetTrainableBlobNames() const override;

virtual std::vector<std::vector<int64_t>> GetTrainableBlobShapes() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

protected:
std::string InputBlobName;
std::string OutputBlobName;
int64_t InputSize;
int64_t OutputSize;
int64_t BatchSize;
std::string WeightFillType;
std::string BiasFillType;
};

}
