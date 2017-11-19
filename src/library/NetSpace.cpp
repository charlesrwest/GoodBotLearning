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

const static std::vector<std::string> allowed_types{"FC", "Scale", "Cast", "Conv", "MaxPool", "CopyCPUToGPU", "CopyGPUToCPU"}; //Could sort and make binary search at some point

return std::find(allowed_types.begin(), allowed_types.end(), type) != allowed_types.end();
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
else if(IsType("Scale", creatorOp) || IsType("Cast", creatorOp) || IsType("CopyCPUToGPU", creatorOp) || IsType("CopyGPUToCPU", creatorOp))
{
//Passthrough, just get input size
return GetBlobShape(GetInputName(creatorOp, 0), netSpace);
}
else if(IsType("Conv", creatorOp))
{
return GetConvOperatorOutputSize(creatorOp, netSpace);
}
else if(IsType("MaxPool", creatorOp))
{
return GetMaxPoolOperatorOutputSize(creatorOp, netSpace);
}
else
{
    SOM_ASSERT(false, "Blob shape determination from " + creatorOp.GetOperatorDef().type() + " has not been implemented yet but our check function thought it had been");
}
}
else
{
    SOM_ASSERT(false, "Blob shape determination from " + creatorOp.GetOperatorDef().type() + " has not been implemented yet");
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
return GetNetwork(rootNetworkName, activeMode, includeEmptyModeOps, caffe2::CPU, netspace); //Device type CPU
}

caffe2::NetDef GoodBot::GetNetwork(const std::string& rootNetworkName, const std::string& activeMode, bool includeEmptyModeOps, int32_t deviceType, const NetSpace& netspace)
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
    network.mutable_device_option()->set_device_type(deviceType);

    return network;
}

std::vector<int64_t> GoodBot::GetFCOperatorOutputSize(const NetOp& op, const NetSpace& netSpace)
{
    std::string input_name = GetInputName(op, 0);
    std::string bias_blob_name = GetInputName(op, 2);

    return {GetBlobShape(input_name, netSpace)[0], GetBlobShape(bias_blob_name, netSpace)[0]};
}

std::vector<int64_t> GoodBot::GetConvOperatorOutputSize(const NetOp& op, const NetSpace& netSpace)
{
    std::string input_name = GetInputName(op, 0);
    std::vector<int64_t> input_shape = GetBlobShape(input_name, netSpace);
    SOM_ASSERT(input_shape.size() == 4, "Unsupported input dimensionality");

    std::string bias_name = GetInputName(op, 2);
    std::vector<int64_t> bias_shape = GetBlobShape(bias_name, netSpace);
    SOM_ASSERT(bias_shape.size() == 1, "Unsupported bias dimensionality");
    int64_t output_depth = bias_shape[0];

    const caffe2::Argument* stride_arg = GetArgument("stride", op.GetOperatorDef());
    SOM_ASSERT(stride_arg != nullptr, "Should not have conv without stride");
    SOM_ASSERT(stride_arg->has_i(), "Stride should be a scalar integer");
    int64_t stride = stride_arg->i();

    const caffe2::Argument* kernel_arg = GetArgument("kernel", op.GetOperatorDef());
    SOM_ASSERT(kernel_arg != nullptr, "Should not have conv without kernel");
    SOM_ASSERT(kernel_arg->has_i(), "kernel should be a scalar integer");
    int64_t kernel = kernel_arg->i();

    const caffe2::Argument* pad_arg = GetArgument("pad", op.GetOperatorDef());
    SOM_ASSERT(pad_arg != nullptr, "Should not have conv without pad");
    SOM_ASSERT(pad_arg->has_i(), "Pad should be a scalar integer");
    int64_t pad = pad_arg->i();

    //Batch size, depth, height width
    int64_t kernel_loss = kernel == 2 ? 0 : ((kernel/2)*2);
    return {input_shape[0], output_depth, (input_shape[2]/stride)+2*pad-kernel_loss, (input_shape[3]/stride)+2*pad-kernel_loss};
}

std::vector<int64_t> GoodBot::GetMaxPoolOperatorOutputSize(const NetOp& op, const NetSpace& netSpace)
{
    std::string input_name = GetInputName(op, 0);
    std::vector<int64_t> input_shape = GetBlobShape(input_name, netSpace);
    SOM_ASSERT(input_shape.size() == 4, "Unsupported input dimensionality");

    const caffe2::Argument* stride_arg = GetArgument("stride", op.GetOperatorDef());
    SOM_ASSERT(stride_arg != nullptr, "Should not have maxpool without stride");
    SOM_ASSERT(stride_arg->has_i(), "Stride should be a scalar integer");
    int64_t stride = stride_arg->i();

    const caffe2::Argument* kernel_arg = GetArgument("kernel", op.GetOperatorDef());
    SOM_ASSERT(kernel_arg != nullptr, "Should not have maxpool without kernel");
    SOM_ASSERT(kernel_arg->has_i(), "kernel should be a scalar integer");
    int64_t kernel = kernel_arg->i();

    const caffe2::Argument* pad_arg = GetArgument("pad", op.GetOperatorDef());
    SOM_ASSERT(pad_arg != nullptr, "Should not have maxpool without pad");
    SOM_ASSERT(pad_arg->has_i(), "Pad should be a scalar integer");
    int64_t pad = pad_arg->i();

    const caffe2::Argument* order_arg = GetArgument("order", op.GetOperatorDef());
    SOM_ASSERT(order_arg != nullptr, "Should not have maxpool without order");
    SOM_ASSERT(order_arg->has_s(), "Order should be a single string");
    SOM_ASSERT(order_arg->s() == "NCHW", "We are only currently supporting NCHW ordering");

    //Batch size, depth, height width
    int64_t kernel_loss = kernel == 2 ? 0 : ((kernel/2)*2);
    return {input_shape[0], input_shape[1], (input_shape[2]/stride)+2*pad-kernel_loss, (input_shape[3]/stride)+2*pad-kernel_loss};
}
