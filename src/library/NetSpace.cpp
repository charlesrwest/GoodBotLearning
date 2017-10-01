#include "NetSpace.hpp"
#include "SOMException.hpp"
#include "UtilityFunctions.hpp"

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

const NetOp* GoodBot::GetCreatorOfBlob(const std::string& blobName, const NetSpace& netspace)
{
const std::map<std::string, NetOp>& netOps = netspace.GetNetOps();

for(const std::pair<std::string, NetOp>& netOp : netOps)
{
if(CreatesBlob(blobName, netOp.second))
{
return &netOp.second;
}
}

return nullptr;
}

std::vector<int64_t> GetBlobShapeFromCreator(const std::string& blobName, const GoodBot::NetOp& creatorOp, const NetSpace& netSpace)
{
SOM_ASSERT(CreatesBlob(blobName, creatorOp), "Can't get shape from op that did not create it");

std::vector<int64_t> result;

const caffe2::Argument* shape_arg = GoodBot::GetArgument("shape", creatorOp.GetOperatorDef());

if(shape_arg != nullptr)
{
//This operator makes the blob and the information can be retrieved directly using the shape arg
for(int64_t dim_index = 0; dim_index < shape_arg->ints_size(); dim_index++)
{
result.emplace_back(shape_arg->ints(dim_index));
}
return result;
}

//The blob is created by a operator def without a shape argument.  This means that the shape can be determined by traversing the workspace/netspace tree, but that is TBD for now
SOM_ASSERT(false, "Blob shape determination from tree has not been implemented yet");
return result;
}

std::vector<int64_t> GoodBot::GetBlobShape(const std::string& blobName, const NetSpace& netSpace)
{
const caffe2::Workspace& workspace = netSpace.GetWorkspace();

if(workspace.HasBlob(blobName))
{
//Blob is in workspace, so get size directly from it
caffe2::TensorCPU tensor = GetTensor(*workspace.GetBlob(blobName));

return tensor.dims();
}

//Not in workspace, check if it is in the netspace
const NetOp* creator_ptr = GoodBot::GetCreatorOfBlob(blobName, netSpace);

SOM_ASSERT(creator_ptr != nullptr, "Blob is not present in workspace or netspace");

return GetBlobShapeFromCreator(blobName, *creator_ptr, netSpace);
}

std::vector<NetOp> GetActiveNetOps(const std::string& rootNetworkName, const std::string& activeMode, const GoodBot::NetSpace& netspace)
{
const std::map<std::string, NetOp>& netops = netspace.GetNetOps();

std::vector<NetOp> active_netops;
for(const std::pair<std::string, NetOp>& netop_pair : netops)
{
if(GoodBot::StringAtStart(rootNetworkName, netop_pair.first))
{
const std::vector<std::string>& active_modes = netop_pair.second.GetActiveModes();
if(std::find(active_modes.begin(), active_modes.end(), activeMode) != active_modes.end())
{
active_netops.emplace_back(netop_pair.second);
}
}
}


return active_netops;
}

std::vector<caffe2::OperatorDef> ConvertToOperatorDef(const std::vector<GoodBot::NetOp>& netOps)
{
std::vector<caffe2::OperatorDef> results;

for(const NetOp& netop : netOps)
{
results.emplace_back(netop.GetOperatorDef());
}

return results;
}

caffe2::NetDef GoodBot::GetNetwork(const std::string& rootNetworkName, const std::string& activeMode, const NetSpace& netspace)
{
std::vector<NetOp> active_netops = GetActiveNetOps(rootNetworkName, activeMode, netspace);

std::cout << "Got " << active_netops.size() << " blob netops of " << netspace.GetNetOps().size() << std::endl;

std::vector<caffe2::OperatorDef> operator_definitions = ConvertToOperatorDef(active_netops);

operator_definitions = ReorderOperatorsToResolveDependencies(operator_definitions, netspace.GetWorkspace().Blobs());

caffe2::NetDef network;
network.set_name(rootNetworkName + "_" + activeMode);

for(const caffe2::OperatorDef& operator_definition : operator_definitions)
{
*network.add_op() = operator_definition; //Add to network
}

return network;
}
