#pragma once

#include "caffe2/core/workspace.h"

namespace GoodBot
{

/**
This class is meant to allow for simple synchronization between blobs in the CPU and blobs in the GPU.  It makes a very simple network that has CopyGPUToCPU and CopyCPUToGPU operators for the designated blobs.
*/
class DataSynchronizer
{
public:
/**
This function creates and registers the required operators with the given workspace.
@param cpuThenGPUBlobNamePairs: A list of pairs of blobs to synchronize
@param workspace: The caffe2 workspace the blobs live in

@throws: This function can throw exceptions
*/
DataSynchronizer(const std::vector<std::pair<std::string, std::string>>& cpuThenGPUBlobNamePairs, caffe2::Workspace& workspace);

/**
This function activates the associated network to trigger a move of the data from the CPU blobs to the GPU blobs.
*/
void MoveCPUDataToGPU();

/**
This function activates the associated network to trigger a move of the data from the GPU blobs to the CPU blobs.
*/
void MoveGPUDataToCPU();

private:
caffe2::NetBase* CPUToGPUNetwork;
caffe2::NetBase* GPUToCPUNetwork;
};

}
