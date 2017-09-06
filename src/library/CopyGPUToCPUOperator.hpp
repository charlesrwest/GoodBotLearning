#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

class CopyGPUToCPUOperator : public ComputeModuleDefinition
{
public:
CopyGPUToCPUOperator(const std::string& name, const std::string& gpuBlobName, const std::string& cpuBlobName);

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string GPUBlobName;
std::string CPUBlobName;
};
















}
