#include "NetSpace.hpp"
#include "SOMException.hpp"
#include "UtilityFunctions.hpp"
#include<algorithm>

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

for(const std::pair<const std::string, NetOp>& netOp : netOps)
{
if(CreatesBlob(blobName, netOp.second))
{
return &netOp.second;
}
}

return nullptr;
}

bool CanResolveBlobShapeFromCreator(const std::string& blobName, const GoodBot::NetOp& creatorOp, const NetSpace& netSpace)
{
SOM_ASSERT(CreatesBlob(blobName, creatorOp), "Can't get shape for blob " + blobName + " from op " + creatorOp.GetName() + " which did not create it");

if(GoodBot::GetArgument("shape", creatorOp.GetOperatorDef()) != nullptr)
{
    //It has a shape argument, so we will assume the blob was made with that shape
    return true;
}

const std::string& type = creatorOp.GetOperatorDef().type();

if(type == "FC")
{
    return true;
}

return false;
}

std::vector<int64_t> GetBlobShapeFromCreator(const std::string& blobName, const GoodBot::NetOp& creatorOp, const NetSpace& netSpace)
{
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

//if FC, find bias blob and grab size from that.
if(CanResolveBlobShapeFromCreator(blobName, creatorOp, netSpace))
{
if(IsType("FC", creatorOp))
{
return GetFCOperatorOutputSize(creatorOp, netSpace);
}
else
{
    SOM_ASSERT(false, "Blob shape determination from " + creatorOp.GetOperatorDef().type() + " has not been implemented yet");
}
}

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

std::vector<NetOp> GoodBot::GetActiveNetOps(const std::string& rootNetworkName, const std::string& activeMode, bool includeEmptyModeOps, const GoodBot::NetSpace& netspace)
{
return GetActiveNetOps(rootNetworkName, std::vector<std::string>{activeMode}, includeEmptyModeOps, netspace);
}

std::vector<NetOp> GoodBot::GetActiveNetOps(const std::string& rootNetworkName, const std::vector<std::string>& activeModes, bool includeEmptyModeOps, const GoodBot::NetSpace& netspace)
{
    const std::map<std::string, NetOp>& netops = netspace.GetNetOps();

    std::vector<NetOp> active_netops;
    for(const std::pair<std::string, NetOp>& netop_pair : netops)
    {
    if(GoodBot::StringAtStart(rootNetworkName, netop_pair.first))
    {
    const std::vector<std::string>& active_modes = netop_pair.second.GetActiveModes();

    if(( includeEmptyModeOps && (active_modes.size() == 0) ) || (activeModes.size() == 0))
    {
       //We have selected to add ops with no active mode listings.
       active_netops.emplace_back(netop_pair.second);
       continue;
    }


    for(const std::string& active_mode : activeModes)
    {
        if(std::find(active_modes.begin(), active_modes.end(), active_mode) != active_modes.end())
        {
        active_netops.emplace_back(netop_pair.second);
        break;
        }
    }
    }
    }

    return active_netops;
}

std::vector<std::string> GoodBot::GetTrainableBlobs(const std::string& networkName, const std::vector<std::string>& activeModes, const GoodBot::NetSpace& netspace)
{
    //Find operators making trainable blobs and get the associated blob names
    std::vector<NetOp> ops = GetActiveNetOps(networkName, activeModes, true, netspace);

    std::vector<std::string> results;
    for(const NetOp& op : ops)
    {
        if(!op.IsTrainable())
        {
            continue;
        }

        for(int64_t output_index = 0; output_index < op.GetOperatorDef().output_size(); output_index++)
        {
            results.emplace_back(op.GetOperatorDef().output(output_index));
        }
    }

    //Remove any duplicate entries
    std::sort( results.begin(), results.end() );
    results.erase( std::unique( results.begin(), results.end() ), results.end() );

    return results;
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

caffe2::NetDef GoodBot::GetNetwork(const std::string& rootNetworkName, const std::string& activeMode, bool includeEmptyModeOps, const NetSpace& netspace)
{
std::vector<NetOp> active_netops = GetActiveNetOps(rootNetworkName, activeMode, includeEmptyModeOps, netspace);

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

std::vector<int64_t> GoodBot::GetFCOperatorOutputSize(const NetOp& op, const NetSpace& netSpace)
{
    std::string input_name = GetInputName(op, 0);
    std::string bias_blob_name = GetInputName(op, 2);

    return {GetBlobShape(input_name, netSpace)[0], GetBlobShape(bias_blob_name, netSpace)[0]};
}
