#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

class CopyCPUToGPUOperator : public ComputeModuleDefinition
{
public:
CopyCPUToGPUOperator(const std::string& name, const std::string& cpuBlobName, const std::string& gpuBlobName);

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

protected:
std::string CPUBlobName;
std::string GPUBlobName;
};

















}
