#include "LabelCrossEntropyOperator.hpp"

using namespace GoodBot;

LabelCrossEntropyOperator::LabelCrossEntropyOperator(const std::string& name, const std::string& softmaxBlobName, const std::string& labelBlobName) : SoftmaxBlobName(softmaxBlobName), LabelBlobName(labelBlobName)
{
SetName(name);
}

std::vector<std::string> LabelCrossEntropyOperator::GetInputBlobNames() const
{
return {SoftmaxBlobName, LabelBlobName};
}

std::vector<std::string> LabelCrossEntropyOperator::GetOutputBlobNames() const
{
if(Mode() == "TRAIN")
{
return {Name()};
}

return {GetInputBlobNames()[0]};
}

std::vector<caffe2::OperatorDef> LabelCrossEntropyOperator::GetNetworkOperators() const
{
if((Mode() == "TRAIN") || (Mode() == "TEST"))
{
std::vector<caffe2::OperatorDef> results;

results.emplace_back();
caffe2::OperatorDef& operator_def = results.back();

operator_def.set_type("LabelCrossEntropy");
operator_def.add_input(SoftmaxBlobName);
operator_def.add_input(LabelBlobName);
operator_def.add_output(GetOutputBlobNames()[0]);

return results;
}
}
