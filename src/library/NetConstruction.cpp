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

void GoodBot::AddFullyConnectedOp(const std::string& opName, const std::string& inputName, const std::string& weightsName, const std::string& biasName, const std::string& outputName, NetSpace& netspace)
{
}

