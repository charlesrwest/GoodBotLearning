#include "NetSpace.hpp"
#include "SOMException.hpp"

using namespace GoodBot;

NetSpace::NetSpace(caffe2::Workspace& workspace) : Workspace(&workspace)
{
}

caffe2::Workspace& NetSpace::GetWorkspace() const
{
return *Workspace;
}

bool NetSpace::HasNetOp(const std::string& netOpName) const
{
return NetOps.count(netOpName) > 0;
}

void NetSpace::AddNetOp(const NetOp& netOp)
{
SOM_ASSERT(!HasNetOp(netOp.GetName()), "NetOp " + netOp.GetName() + " already exists." );

NetOps.emplace(netOp.GetName(), netOp);
}

const NetOp& NetSpace::GetNetOp(const std::string& netOpName) const
{
SOM_ASSERT(HasNetOp(netOpName), "NetOp " + netOpName + " not found.");

return NetOps.at(netOpName);
}

const std::map<std::string, NetOp>& NetSpace::GetNetOps() const
{
return NetOps;
}

std::vector<int64_t> GoodBot::GetBlobShape(const std::string& blobName, const NetSpace& netSpace)
{
std::vector<int64_t> result;

const caffe2::Workspace& workspace = netSpace.GetWorkspace();

if(workspace.HasBlob(blobName))
{
//Blob is in workspace, so get size directly from it
caffe2::TensorCPU tensor = GetTensor(*workspace.GetBlob(blobName));

result = tensor.dims();
}
else
{
//Check if it is in the netspace

}

return result;
}
