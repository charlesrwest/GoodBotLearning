#include "CopyCPUToGPUOperator.hpp"


using namespace GoodBot;


CopyCPUToGPUOperator::CopyCPUToGPUOperator(const std::string& name, const std::string& cpuBlobName, const std::string& gpuBlobName) : CPUBlobName(cpuBlobName), GPUBlobName(gpuBlobName)
{
SetName(name);
}

std::vector<caffe2::OperatorDef> CopyCPUToGPUOperator::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("CopyCPUToGPU");
operatorDef.add_input(CPUBlobName);
operatorDef.add_output(GPUBlobName);

return results;
}
