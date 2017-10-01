#include "NetConstruction.hpp"
#include "SOMException.hpp"

using namespace GoodBot;

Arg::Arg(const std::string& name, double value) : Arg(name, (float) value)
{
}

Arg::Arg(const std::string& name, float value) : Name(name), IsScalar(true), Floats{value}
{
}

Arg::Arg(const std::string& name, int value) : Arg(name, (int64_t) value)
{
}

Arg::Arg(const std::string& name, int64_t value) : Name(name), IsScalar(true), Ints{value}
{
}

Arg::Arg(const std::string& name, const std::string& value) : Name(name), IsScalar(true), Strings{value}
{
}

Arg::Arg(const std::string& name, const std::initializer_list<double>& value) : Name(name)
{
Floats.reserve(value.size());
for(double scalar : value)
{
Floats.emplace_back(scalar);
}
}

Arg::Arg(const std::string& name, const std::initializer_list<float>& value) : Name(name)
{
Floats.reserve(value.size());
for(float scalar : value)
{
Floats.emplace_back(scalar);
}
}

Arg::Arg(const std::string& name, const std::initializer_list<int>& value) : Name(name)
{
Ints.reserve(value.size());
for(int scalar : value)
{
Ints.emplace_back(scalar);
}
}

Arg::Arg(const std::string& name, const std::initializer_list<int64_t>& value) : Name(name)
{
Ints.reserve(value.size());
for(int scalar : value)
{
Ints.emplace_back(scalar);
}
}

Arg::Arg(const std::string& name, const std::initializer_list<std::string>& value) : Name(name)
{
Strings.reserve(value.size());
for(const std::string& string : value)
{
Strings.emplace_back(string);
}
}

Arg::Arg(const std::string& name, const std::vector<double>& value) : Name(name)
{
Floats.reserve(value.size());
for(double scalar : value)
{
Floats.emplace_back(scalar);
}
}

Arg::Arg(const std::string& name, const std::vector<float>& value) : Name(name), Floats(value)
{
}

Arg::Arg(const std::string& name, const std::vector<int>& value) : Name(name)
{
Ints.reserve(value.size());
for(int scalar : value)
{
Ints.emplace_back(scalar);
}
}

Arg::Arg(const std::string& name, const std::vector<int64_t>& value) : Name(name), Ints(value)
{
}

Arg::Arg(const std::string& name, const std::vector<std::string>& value) : Name(name), Strings(value)
{
}

Arg::operator caffe2::Argument() const
{
caffe2::Argument value;

value.set_name(Name);

if(IsScalar)
{
if(Floats.size() > 0)
{
value.set_f(Floats[0]);
}
else if(Ints.size() > 0)
{
value.set_i(Ints[0]);
}
else if(Strings.size() > 0)
{
value.set_s(Strings[0]);
}
else
{
SOM_ASSERT(false, "Should never reach here" );
}
}
else
{
for(float scalar : Floats)
{
value.add_floats(scalar);
}

for(int64_t scalar : Ints)
{
value.add_ints(scalar);
}

for(const std::string& string : Strings)
{
value.add_strings(string);
}
}

return value;
}

caffe2::OperatorDef GoodBot::CreateOpDef(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const std::string& type, const std::vector<GoodBot::Arg>& arguments)
{
caffe2::OperatorDef result;

result.set_name(name);

for(const std::string& input : inputs)
{
result.add_input(input);
}

for(const std::string& output : outputs)
{
result.add_output(output);
}

result.set_type(type);

for(const GoodBot::Arg& argument : arguments)
{
(*result.add_arg()) = argument;
}

return result;
}

void GoodBot::AddXavierOp(const std::string& opName, const std::string& outputName, const std::vector<int64_t>& blobDimensions, const std::vector<std::string>& activeModes, bool isTrainable, NetSpace& netspace)
{
NetOp op(GoodBot::CreateOpDef(opName, {}, {outputName}, "XavierFill", {{"shape", blobDimensions}}), activeModes, isTrainable);

netspace.AddNetOp(op);
}

void GoodBot::AddFullyConnectedOp(const std::string& opName, const std::string& inputName, const std::string& weightsName, const std::string& biasName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
NetOp op(GoodBot::CreateOpDef(opName, {inputName, weightsName, biasName}, {outputName}, "FC", {}), activeModes, false);

netspace.AddNetOp(op);
}

void GoodBot::AddReluOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "Relu", {}), activeModes, false);

netspace.AddNetOp(op);
}

void GoodBot::AddTanhOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "Tanh", {}), activeModes, false);

netspace.AddNetOp(op);
}

void GoodBot::AddFullyConnectedModule(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t outputSize, const std::string& weightFillType, const std::string& biasFillType, NetSpace& netspace)
{
std::vector<int64_t> inputBlobShape = GetBlobShape(inputName, netspace);
SOM_ASSERT(inputBlobShape.size() >= 2, "Need batch size as part of blob dimension");

int64_t batch_size = inputBlobShape[0];
int64_t inputSize = 1;
for(int64_t inputShapeIndex = 1; inputShapeIndex < inputBlobShape.size(); inputShapeIndex++)
{
inputSize *= inputBlobShape[inputShapeIndex];
}

//Add initialization operators
SOM_ASSERT(weightFillType == "XavierFill", "Unsupported fill type");
std::string weight_name = opName+"_weight_fill";
AddXavierOp(weight_name, weight_name, {outputSize, inputSize}, {"INIT"}, true, netspace);

SOM_ASSERT(biasFillType == "ConstantFill", "Unsupported fill type");
std::string bias_name = opName+"_bias_fill";
AddConstantFillOp(bias_name, bias_name, 0.0f, caffe2::TensorProto::FLOAT, {outputSize}, {"INIT"}, true, netspace);

//Add operators
std::string fc_name = opName + "_fc";
AddFullyConnectedOp(fc_name, inputName, weight_name, bias_name, outputName, {}, netspace);
}

void GoodBot::AddFullyConnectedModuleWithActivation(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t outputSize, const std::string& activationType, const std::string& weightFillType, const std::string& biasFillType, NetSpace& netspace)
{
AddFullyConnectedModule(opName+"_m", inputName, outputName, outputSize, weightFillType, biasFillType, netspace);

if(activationType == "Relu")
{
AddReluOp(opName+"_relu", outputName, outputName, {}, netspace);
}
else if(activationType == "Tanh")
{
AddTanhOp(opName+"_tanh", outputName, outputName, {}, netspace);
}
else
{
SOM_ASSERT(false, "Unsupported activation type");
}
}