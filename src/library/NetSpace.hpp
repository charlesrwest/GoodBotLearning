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
This function determines the shape of the blob with the given name by examining existing blobs in the workspace/netspace.
@param blobName: The name of the blob to examine
@return: The shape of the blob
*/
std::vector<int64_t> GetBlobShape(const std::string& blobName, const NetSpace& netSpace);












}
