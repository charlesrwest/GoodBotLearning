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

void AddFullyConnectedOp(const std::string& opName, const std::string& inputName, const std::string& weightsName, const std::string& biasName, const std::string& outputName, NetSpace& netspace);

//void AddReluOp(const std::string& opName, const std::string& inputName, const std::string& outputName, NetSpace& netspace);

//void AddTanhOp(const std::string& opName, const std::string& inputName, const std::string& outputName, NetSpace& netspace);

//void AddFullyConnectedModule(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t outputSize, const std::string& weightFillType = "XavierFill", const std::string& biasFillType = "ConstantFill");

template<class DType>
void AddConstantFillOp(const std::string& opName, const std::string& outputName, DType value,  caffe2::TensorProto::DataType type, const std::vector<int64_t>& blobDimensions, const std::vector<std::string>& activeModes, bool isTrainable, NetSpace& netspace)
{
NetOp op(GoodBot::CreateOpDef(opName, {}, {outputName}, "Constant", {{"shape", blobDimensions}, {"value", value}, {"dtype", type}}), activeModes, isTrainable);

netspace.AddNetOp(op);
}







}
