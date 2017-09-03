#include "VGG16.hpp"
#include "UtilityFunctions.hpp"
#include "VGGConvolutionModule.hpp"
#include "SoftMaxLayerDefinition.hpp"
#include "FullyConnectedOperator.hpp"
#include "FullyConnectedModuleDefinition.hpp"
#include "CastModule.hpp"
#include "ScaleModule.hpp"
#include<iostream>

using namespace GoodBot;

VGG16::VGG16(const VGG16Parameters& inputParameters)
{
SetName(inputParameters.Name);

std::string casted_input_blob_name = inputParameters.InputBlobName + "_casted";

CastModuleParameters cast_parameters;
cast_parameters.name = Name() + "_cast";
cast_parameters.InputBlobName = inputParameters.InputBlobName;
cast_parameters.TargetDataType = caffe2::TensorProto_DataType_FLOAT;
cast_parameters.OutputBlobName = casted_input_blob_name;

AddModule(*(new CastModule(cast_parameters)));

ScaleModuleParameters scale_parameters;
scale_parameters.name = Name() + "_scale";
scale_parameters.InputBlobName = casted_input_blob_name;
scale_parameters.OutputBlobName = casted_input_blob_name;
scale_parameters.Scale = 1.0 / 256.0;

AddModule(*(new ScaleModule(scale_parameters)));

//Add cast operator to convert from uchar to float
//predictionNetwork.AddCastOp("data_uint8", "data", caffe2::TensorProto_DataType_FLOAT);
//predictionNetwork.AddScaleOp("data", "data", 1.0 / 256.0);

VGGConvolutionModuleParameters parameters;

struct ConvolutionParameters
{
std::string LayerName;
int64_t NumberOfConvolutions;
int64_t OutputDepth;
};

std::vector<ConvolutionParameters> conv_parameters
{
{Name() + "_vgg_1", 2, 64},
{Name() + "_vgg_2", 2, 128},
{Name() + "_vgg_3", 3, 256},
{Name() + "_vgg_4", 3, 512},
{Name() + "_vgg_5", 3, 512}
};

for(int64_t conv_index = 0; conv_index < conv_parameters.size(); conv_index++)
{
const ConvolutionParameters& parameters = conv_parameters[conv_index];

AddModule(*(new VGGConvolutionModule(
VGGConvolutionModuleParameters
{
parameters.LayerName,
conv_index == 0 ? casted_input_blob_name : modules.back()->GetOutputBlobNames()[0],
parameters.LayerName,
conv_index == 0 ? 3 : conv_parameters[conv_index-1].OutputDepth,
parameters.OutputDepth,
parameters.NumberOfConvolutions
}
)));
}


//Add 2 fully connected layers with activations
AddModule(*(new FullyConnectedModuleDefinition(
modules.back()->GetOutputBlobNames()[0],
std::vector<int64_t>{4096, 4096},
Name() + "_fc_relu",
32768)));

//Add a fully connected layer with a 1 hot output
AddModule(*(new FullyConnectedOperator(
Name() + "_fc1",
modules.back()->GetOutputBlobNames()[0],
Name() + "_fc1",
4096,
5
)));

AddModule(*(new SoftMaxLayerDefinition(
{
modules.back()->GetOutputBlobNames()[0],
Name() + "_soft_max",
inputParameters.TrainingExpectedOutputBlobName,
inputParameters.TestExpectedOutputBlobName
})));

LastModuleIndex = modules.size() - 1;
}

std::vector<std::string> VGG16::GetInputBlobNames() const
{
return modules.front()->GetInputBlobNames();
}

std::vector<std::string> VGG16::GetOutputBlobNames() const
{
return modules[LastModuleIndex]->GetOutputBlobNames();
}
