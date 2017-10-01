#pragma once

#include<map>
#include<string>
#include "caffe2/core/workspace.h"
#include "NetOp.hpp"

namespace GoodBot
{

class NetSpace
{
public:
NetSpace(caffe2::Workspace& workspace);

caffe2::Workspace& GetWorkspace() const;
void AddNetOp(const NetOp& netOp);
bool HasNetOp(const std::string& netOpName) const;
const NetOp& GetNetOp(const std::string& netOpName) const;
const std::map<std::string, NetOp>& GetNetOps() const;

private:
caffe2::Workspace* Workspace;
std::map<std::string, NetOp> NetOps;
};

/**
Create a caffe2 definition of a netop network from a root name.
@param rootNetworkName: The root name of all networks in this network.
@param activeMode: The mode that is active (determines which ops are pulled in)
@param netspace: The netspace to pull operators from
*/
caffe2::NetDef GetNetwork(const std::string& rootNetworkName, const std::string& activeMode, const NetSpace& netspace); 

//Returns null if the blob is not found
const NetOp* GetCreatorOfBlob(const std::string& blobName, const NetSpace& netspace);

/**
This function determines the shape of the blob with the given name by examining existing blobs in the workspace/netspace.
@param blobName: The name of the blob to examine
@return: The shape of the blob
*/
std::vector<int64_t> GetBlobShape(const std::string& blobName, const NetSpace& netSpace);












}