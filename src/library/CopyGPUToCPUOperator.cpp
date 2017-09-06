#include "CopyGPUToCPUOperator.hpp"

using namespace GoodBot;


CopyGPUToCPUOperator::CopyGPUToCPUOperator(const std::string& name, const std::string& gpuBlobName, const std::string& cpuBlobName) : GPUBlobName(gpuBlobName), CPUBlobName(cpuBlobName)
{
SetName(name);
}

std::vector<caffe2::OperatorDef> CopyGPUToCPUOperator::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("CopyGPUToCPU");
operatorDef.add_input(GPUBlobName);
operatorDef.add_output(CPUBlobName);

return results;
}

