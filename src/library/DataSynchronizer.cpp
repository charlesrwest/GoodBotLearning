#include "DataSynchronizer.hpp"
#include "CompositeComputeModuleDefinition.hpp"
#include "CopyCPUToGPUOperator.hpp"
#include "CopyGPUToCPUOperator.hpp"

using namespace GoodBot;

/**
This function creates and registers the required operators with the given workspace.
@param cpuThenGPUBlobNamePairs: A list of pairs of blobs to synchronize
@param workspace: The caffe2 workspace the blobs live in

@throws: This function can throw exceptions
*/
DataSynchronizer::DataSynchronizer(const std::vector<std::pair<std::string, std::string>>& cpuThenGPUBlobNamePairs, caffe2::Workspace& workspace)
{
//Create network to move CPU -> GPU
{
CompositeComputeModuleDefinition cpu_to_gpu_module;
cpu_to_gpu_module.SetName("DataSynchronizerToGPU");
for(const std::pair<std::string, std::string>& cpu_blob_gpu_blob : cpuThenGPUBlobNamePairs)
{
cpu_to_gpu_module.AddModule(*(new GoodBot::CopyCPUToGPUOperator("", cpu_blob_gpu_blob.first, cpu_blob_gpu_blob.second)));
}

cpu_to_gpu_module.SetMode("DEPLOY");
caffe2::NetDef cpu_to_gpu_network_definition = cpu_to_gpu_module.GetNetwork(workspace.Blobs());
cpu_to_gpu_network_definition.mutable_device_option()->set_device_type(caffe2::CUDA);
CPUToGPUNetwork = workspace.CreateNet(cpu_to_gpu_network_definition);
}

//Create network to move GPU -> CPU
{
CompositeComputeModuleDefinition gpu_to_cpu_module;
gpu_to_cpu_module.SetName("DataSynchronizerToCPU");
for(const std::pair<std::string, std::string>& cpu_blob_gpu_blob : cpuThenGPUBlobNamePairs)
{
gpu_to_cpu_module.AddModule(*(new GoodBot::CopyCPUToGPUOperator("", cpu_blob_gpu_blob.second, cpu_blob_gpu_blob.first)));
}

gpu_to_cpu_module.SetMode("DEPLOY");
caffe2::NetDef gpu_to_cpu_network_definition = gpu_to_cpu_module.GetNetwork(workspace.Blobs());
gpu_to_cpu_network_definition.mutable_device_option()->set_device_type(caffe2::CUDA);
GPUToCPUNetwork = workspace.CreateNet(gpu_to_cpu_network_definition);
}
}

/**
This function activates the associated network to trigger a move of the data from the CPU blobs to the GPU blobs.
*/
void DataSynchronizer::MoveCPUDataToGPU()
{
CPUToGPUNetwork->Run();
}

/**
This function activates the associated network to trigger a move of the data from the GPU blobs to the CPU blobs.
*/
void DataSynchronizer::MoveGPUDataToCPU()
{
GPUToCPUNetwork->Run();
}


