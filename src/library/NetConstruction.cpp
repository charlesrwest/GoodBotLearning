#include "NetConstruction.hpp"
#include "SOMException.hpp"
#include "UtilityFunctions.hpp"

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

void GoodBot::AddSquaredL2DistanceOp(const std::string& opName, const std::string& firstInputName, const std::string& secondInputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {firstInputName, secondInputName}, {outputName}, "SquaredL2Distance", {}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddNetworkGradientLoopBack(const std::string& opName, const std::string& inputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {MakeGradientOperatorBlobName(inputName)}, "ConstantFill", {{"value", 1.0f}, {"dtype", caffe2::TensorProto::FLOAT}}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddAveragedLossOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "AveragedLoss", {}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddGradientOperators(const NetOp& op, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
std::vector<caffe2::OperatorDef> gradient_operators = GetGradientOperatorsFromOperator(op.GetOperatorDef());

for(const caffe2::OperatorDef& opDef : gradient_operators)
{
netspace.AddNetOp(NetOp(opDef, activeModes, false));
}
}

void GoodBot::AddGradientOperators(const std::string& networkName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
std::vector<NetOp> ops = GetActiveNetOps(networkName, activeModes, true, netspace);

for(const NetOp& op : ops)
{
    if(!MakesGradients(op))
    {
        continue;
    }

    std::vector<caffe2::OperatorDef> gradient_operators = GetGradientOperatorsFromOperator(op.GetOperatorDef());

    for(int64_t gradient_def_index = 0; gradient_def_index < gradient_operators.size(); gradient_def_index++)
    {
        caffe2::OperatorDef opDef = gradient_operators[gradient_def_index];
        (*opDef.mutable_name()) = op.GetName() + "_grad_" + std::to_string(gradient_def_index);

        netspace.AddNetOp(NetOp(opDef, activeModes, false));
    }
}
}

void GoodBot::AddAdamOp(const std::string& opName, const std::string& blobToUpdateName, const std::string& learningRateName, const std::string& iteratorName, const std::string& moment1Name, const std::string& moment2Name, const std::vector<std::string>& activeModes, NetSpace& netspace, float beta1, float beta2, float epsilon)
{
    NetOp op(
                GoodBot::CreateOpDef(opName,
    {blobToUpdateName, moment1Name, moment2Name, MakeGradientOperatorBlobName(blobToUpdateName), learningRateName, iteratorName},
    {blobToUpdateName, moment1Name, moment2Name},
                                     "Adam", {{"beta1", beta1}, {"beta2", beta2}, {"epsilon", epsilon}}), activeModes, false);

    netspace.AddNetOp(op);
}


void GoodBot::AddAdamSolvers(const std::string& networkName, NetSpace& netspace, float beta1, float beta2, float epsilon, float learningRate)
{
    std::vector<std::string> trainable_blobs = GetTrainableBlobs(networkName, {}, netspace);

    if(trainable_blobs.size() == 0)
    {
        //Nothing to train, so do nothing
        return;
    }

    //Make iterator blob/operator
    std::string iter_blob_name = networkName+"_adam_iteration_count";
    AddConstantFillOp(iter_blob_name, iter_blob_name, (int64_t) 0, caffe2::TensorProto::INT64, {1}, {"INIT"}, false, caffe2::CPU, netspace ); //Force iter blob CPU
    AddIterOp(networkName+"_adam_iter", networkName+"_adam_iter", {"TRAIN"}, netspace);

    //Make learning rate blob
    std::string learning_rate_blob_name = networkName+"_adam_learning_rate";
    AddConstantFillOp(learning_rate_blob_name, learning_rate_blob_name, learningRate,  caffe2::TensorProto::FLOAT, {1}, {"INIT"}, false, netspace);

    //Make moment blobs/adam operators
    for(const std::string& trainable_blob : trainable_blobs)
    {
        std::string momentum_1_name = trainable_blob + "_moment_1";
        std::string momentum_2_name = trainable_blob + "_moment_2";

        AddConstantFillOp(momentum_1_name, momentum_1_name, 0.0f,  caffe2::TensorProto::FLOAT, GetBlobShape(trainable_blob, netspace), {"INIT"}, false, netspace);
        AddConstantFillOp(momentum_2_name, momentum_2_name, 0.0f,  caffe2::TensorProto::FLOAT, GetBlobShape(trainable_blob, netspace), {"INIT"}, false, netspace);
        AddAdamOp(trainable_blob + "_solver", trainable_blob, learning_rate_blob_name, iter_blob_name, momentum_1_name, momentum_2_name, {"TRAIN"}, netspace, beta1, beta2, epsilon);
    }
}

void GoodBot::AddIterOp(const std::string& opName, const std::string& inputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {inputName}, "Iter", {}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddSoftMaxOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "Softmax", {}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddLabelCrossEntropyOp(const std::string& opName, const std::string& inputName, const std::string& expectedInputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName, expectedInputName}, {outputName}, "LabelCrossEntropy", {}), activeModes, false);

    netspace.AddNetOp(op);
}

//Often "UINT8" -> "FLOAT"
void GoodBot::AddCastOp(const std::string& opName, const std::string& inputName, const std::string& inputTypeString, const std::string& outputName, const std::string& outputTypeString, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "Cast", {{"to", outputTypeString}, {"from_type", inputTypeString}}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddScaleOp(const std::string& opName, const std::string& inputName, const std::string& outputName, float scale, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "Scale", {{"scale", scale}}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddConvOp(const std::string& opName, const std::string& inputName, const std::string& weightsName, const std::string& biasName, const std::string& outputName, int64_t stride, int64_t paddingSize, int64_t kernelSize, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName, weightsName, biasName}, {outputName}, "Conv", {{"stride", stride}, {"pad", paddingSize}, {"kernel", kernelSize}}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddConvModule(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t outputDepth, int64_t stride, int64_t paddingSize, int64_t kernelSize, const std::string& weightFillType, const std::string& biasFillType, NetSpace& netspace)
{
    std::vector<int64_t> input_shape = GetBlobShape(inputName, netspace);
    SOM_ASSERT(input_shape.size() > 2, "Unsupported image shape");

    int64_t input_depth = input_shape[1];

    SOM_ASSERT(weightFillType == "XavierFill", "Unsupported fill type: " + weightFillType);
    std::string weight_name = opName+"_weight_fill";
    AddXavierOp(weight_name, weight_name, {outputDepth, input_depth, kernelSize, kernelSize}, {"INIT"}, true, netspace);

    SOM_ASSERT(biasFillType == "ConstantFill", "Unsupported fill type: " + biasFillType);
    std::string bias_name = opName+"_bias_fill";
    AddConstantFillOp(bias_name, bias_name, 0.0f, caffe2::TensorProto::FLOAT, {outputDepth}, {"INIT"}, true, netspace);

    AddConvOp(opName, inputName, weight_name, bias_name, outputName, stride, paddingSize, kernelSize, std::vector<std::string>{}, netspace);
}

void GoodBot::AddConvModuleWithActivation(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t outputDepth, int64_t stride, int64_t paddingSize, int64_t kernelSize, const std::string& activationType, const std::string& weightFillType, const std::string& biasFillType, NetSpace& netspace)
{
    AddConvModule(opName, inputName, outputName, outputDepth, stride, paddingSize, kernelSize, weightFillType, biasFillType, netspace);

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

void GoodBot::AddMaxPoolOp(const std::string& opName, const std::string& inputName, const std::string& outputName, int64_t stride, int64_t paddingSize, int64_t kernelSize, const std::string& imageOrder, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "MaxPool", {{"stride", stride}, {"pad", paddingSize}, {"kernel", kernelSize}, {"order", imageOrder}}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddCopyCPUToGPU(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "CopyCPUToGPU", {}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddCopyGPUToCPU(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName}, {outputName}, "CopyGPUToCPU", {}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddSpatialBNOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::string& scaleName, const std::string& biasName, const std::string& meanName, const std::string& varianceName, const std::string& savedMeanName, const std::string& savedVarianceName, float momentum, float epsilon, const std::string& dataOrder, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName, scaleName, biasName, meanName, varianceName}, {outputName, meanName, varianceName, savedMeanName, savedVarianceName}, "SpatialBN", {{"is_test", false}, {"momentum", momentum}, {"epsilon", epsilon}, {"order", dataOrder}}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddSpatialBNOp(const std::string& opName, const std::string& inputName, const std::string& outputName, const std::string& scaleName, const std::string& biasName, const std::string& meanName, const std::string& varianceName, float momentum, float epsilon, const std::string& dataOrder, const std::vector<std::string>& activeModes, NetSpace& netspace)
{
    NetOp op(GoodBot::CreateOpDef(opName, {inputName, scaleName, biasName, meanName, varianceName}, {outputName}, "SpatialBN", {{"is_test", true}, {"momentum", momentum}, {"epsilon", epsilon}, {"order", dataOrder}}), activeModes, false);

    netspace.AddNetOp(op);
}

void GoodBot::AddSpatialBNModule(const std::string& opName, const std::string& inputName, const std::string& outputName, float momentum, float epsilon, const std::string& dataOrder, const std::vector<std::string>& trainActiveModes, const std::vector<std::string>& testActiveModes, NetSpace& netspace)
{
    std::vector<int64_t> input_shape = GetBlobShape(inputName, netspace);

    //Initialization
    std::string scale_name = opName+"_scale";
    AddXavierOp(scale_name, scale_name, {input_shape[1]}, {"INIT"}, true, netspace);

    std::string bias_name = opName+"_bias";
    AddConstantFillOp(bias_name, bias_name, 0.0f, caffe2::TensorProto::FLOAT, {input_shape[1]}, {"INIT"}, true, netspace);

    std::string mean_name = opName+"_mean";
    AddXavierOp(mean_name, mean_name, {input_shape[1]}, {"INIT"}, false, netspace);

    std::string variance_name = opName+"_variance";
    AddXavierOp(variance_name, variance_name, {input_shape[1]}, {"INIT"}, false, netspace);

    std::string saved_mean_name = opName+"_saved_mean";
    AddXavierOp(saved_mean_name, saved_mean_name, {input_shape[1]}, {"INIT"}, false, netspace);

    std::string saved_variance_name = opName+"_saved_variance";
    AddXavierOp(saved_variance_name, saved_variance_name, {input_shape[1]}, {"INIT"}, false, netspace);

    //Add train version
    AddSpatialBNOp(opName + "_train", inputName, outputName, scale_name, bias_name, mean_name, variance_name, saved_mean_name, saved_variance_name, momentum, epsilon, dataOrder, trainActiveModes, netspace);

    //Add test version
    AddSpatialBNOp(opName + "_test", inputName, outputName, scale_name, bias_name, mean_name, variance_name, momentum, epsilon, dataOrder, testActiveModes, netspace);
}
