#pragma once

#include "NetSpace.hpp"

namespace GoodBot
{

class Arg
{
public:
Arg(const std::string& name, double value);
Arg(const std::string& name, float value);
Arg(const std::string& name, int value);
Arg(const std::string& name, int64_t value);
Arg(const std::string& name, const std::string& value);

Arg(const std::string& name, const std::initializer_list<double>& value);
Arg(const std::string& name, const std::initializer_list<float>& value);
Arg(const std::string& name, const std::initializer_list<int>& value);
Arg(const std::string& name, const std::initializer_list<int64_t>& value);
Arg(const std::string& name, const std::initializer_list<std::string>& value);

Arg(const std::string& name, const std::vector<double>& value);
Arg(const std::string& name, const std::vector<float>& value);
Arg(const std::string& name, const std::vector<int>& value);
Arg(const std::string& name, const std::vector<int64_t>& value);
Arg(const std::string& name, const std::vector<std::string>& value);

operator caffe2::Argument() const;

private:
std::string Name;
bool IsScalar = false;
std::vector<float> Floats;
std::vector<int64_t> Ints;
std::vector<std::string> Strings;
};

caffe2::OperatorDef CreateOpDef(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const std::string& type, const std::vector<GoodBot::Arg>& arguments);

template<class DType>
void AddConstantFillOp(const std::string& opName, const std::string& outputName, DType value, caffe2::TensorProto::DataType type, const std::vector<int64_t>& blobDimensions, const std::vector<std::string>& activeModes, bool isTrainable, NetSpace& netspace);

void AddXavierOp(const std::string& opName, const std::string& outputName, const std::vector<int64_t>& blobDimensions, const std::vector<std::string>& activeModes, bool isTrainable, NetSpace& netspace);

void AddFullyConnectedOp(const std::string& opName, const std::string& inputName, const std::string& weightsName, const std::string& biasName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddReluOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddTanhOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddFullyConnectedModule(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t outputSize, const std::string& weightFillType, const std::string& biasFillType, NetSpace& netspace);

void AddFullyConnectedModuleWithActivation(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t outputSize, const std::string& activationType, const std::string& weightFillType, const std::string& biasFillType, NetSpace& netspace);

template<class DType>
void AddConstantFillOp(const std::string& opName, const std::string& outputName, DType value,  caffe2::TensorProto::DataType type, const std::vector<int64_t>& blobDimensions, const std::vector<std::string>& activeModes, bool isTrainable, NetSpace& netspace)
{
NetOp op(GoodBot::CreateOpDef(opName, {}, {outputName}, "ConstantFill", {{"shape", blobDimensions}, {"value", value}, {"dtype", type}}), activeModes, isTrainable);

netspace.AddNetOp(op);
}

void AddSquaredL2DistanceOp(const std::string& opName, const std::string& firstInputName, const std::string& secondInputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddNetworkGradientLoopBack(const std::string& opName, const std::string& inputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddAveragedLossOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddGradientOperators(const NetOp& op, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddGradientOperators(const std::string& networkName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddAdamOp(const std::string& opName, const std::string& blobToUpdateName, const std::string& learningRateName, const std::string& iteratorName, const std::string& moment1Name, const std::string& moment2Name, const std::vector<std::string>& activeModes, NetSpace& netspace, float beta1 = .9, float beta2 = .999, float epsilon = 1e-5);

void AddAdamSolvers(const std::string& networkName, NetSpace& netspace, float beta1 = .9, float beta2 = .999, float epsilon = 1e-5, float learningRate = -.001);

void AddIterOp(const std::string& opName, const std::string& inputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddSoftMaxOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

void AddLabelCrossEntropyOp(const std::string& opName, const std::string& inputName, const std::string& expectedInputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace);

}
