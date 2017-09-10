#include "AddOperator.hpp"

using namespace GoodBot;

AddOperator::AddOperator(const std::string& name, const std::string& inputBlobName, float scalarValue) : LHSBlobName(inputBlobName), ScalarValue(scalarValue), GenerateBlob(true), Broadcasting(true)
{
SetName(name);
RHSBlobName = Name() + "_scalar_value";
}

AddOperator::AddOperator(const std::string& name, const std::string& lhsBlobName, const std::string& rhsBlobName, bool broadcasting) : LHSBlobName(lhsBlobName), RHSBlobName(rhsBlobName), Broadcasting(broadcasting), GenerateBlob(false)
{
SetName(name);
}

std::vector<std::string> AddOperator::GetInputBlobNames() const
{
return {LHSBlobName, RHSBlobName};
}

std::vector<std::string> AddOperator::GetOutputBlobNames() const
{
return {Name()};
}

std::vector<caffe2::OperatorDef> AddOperator::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& operatorDef = results.back();

operatorDef.set_type("Add");
operatorDef.add_input(LHSBlobName);
operatorDef.add_input(RHSBlobName);
operatorDef.add_output(Name());

if(Broadcasting)
{
caffe2::Argument& value = *operatorDef.add_arg();
value.set_name("broadcast");
value.set_i(1.0);
}

return results;
}

std::vector<caffe2::OperatorDef> AddOperator::GetNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> result;

if(GenerateBlob)
{
//Make scalar value for addition
result.emplace_back();
caffe2::OperatorDef& scalar_operator = result.back();
scalar_operator.set_type("ConstantFill");

caffe2::Argument& shape = *scalar_operator.add_arg();
shape.set_name("shape");
shape.add_ints(1);
scalar_operator.add_output(RHSBlobName);

caffe2::Argument& value = *scalar_operator.add_arg();
value.set_name("value");
value.set_f(ScalarValue);
}

return result;
}
