#include "NetOp.hpp"

using namespace GoodBot;

NetOp::NetOp(const caffe2::OperatorDef operatorDef, const std::vector<std::string>& activeModes, bool trainable) : OperatorDef(operatorDef), ActiveModes(activeModes), Trainable(trainable)
{
}

const std::string& NetOp::GetName() const
{
return OperatorDef.name();
}

const caffe2::OperatorDef& NetOp::GetOperatorDef() const
{
return OperatorDef;
}

const std::vector<std::string>& NetOp::GetActiveModes() const
{
return ActiveModes;
}

bool NetOp::IsTrainable() const
{
return Trainable;
}

caffe2::TensorCPU GoodBot::GetTensor(const caffe2::Blob &blob) 
{
return blob.Get<caffe2::TensorCPU>();
}

int64_t GoodBot::GetNumInput(const NetOp& op)
{
return op.GetOperatorDef().input_size();
}

int64_t GoodBot::GetNumOutput(const NetOp& op)
{
return op.GetOperatorDef().output_size();
}

bool GoodBot::HasOutputWithoutInput(const NetOp& op)
{
return (GetNumInput(op) == 0) && (GetNumOutput(op) > 0);
}

bool GoodBot::HasInputBlob(const std::string& blobName, const NetOp& op)
{
const caffe2::OperatorDef& op_def = op.GetOperatorDef();

for(int64_t name_index = 0; name_index < op_def.input_size(); name_index++)
{
if(op_def.input(name_index) == blobName)
{
return true;
}
}

return false;
}

bool GoodBot::HasOutputBlob(const std::string& blobName, const NetOp& op)
{
const caffe2::OperatorDef& op_def = op.GetOperatorDef();

for(int64_t name_index = 0; name_index < op_def.output_size(); name_index++)
{
if(op_def.output(name_index) == blobName)
{
return true;
}
}

return false;
}

bool GoodBot::CreatesBlob(const std::string& blobName, const NetOp& op)
{
return (!HasInputBlob(blobName, op)) && HasOutputBlob(blobName, op);
}

const caffe2::Argument* GoodBot::GetArgument(const std::string& argumentName, const caffe2::OperatorDef& opDef)
{
for(int64_t arg_index = 0; arg_index < opDef.arg_size(); arg_index++)
{
const caffe2::Argument& arg = opDef.arg(arg_index);

if(arg.name() == argumentName)
{
return &arg;
}
}

return nullptr;
}
