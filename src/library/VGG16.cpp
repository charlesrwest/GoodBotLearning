#include "VGG16.hpp"
#include "UtilityFunctions.hpp"
#include "VGGConvolutionModule.hpp"
#include "SoftMaxLayerDefinition.hpp"
#include "FullyConnectedOperator.hpp"
#include "FullyConnectedModuleDefinition.hpp"
#include<iostream>

using namespace GoodBot;

VGG16::VGG16(const VGG16Parameters& inputParameters)
{
SetName(inputParameters.Name);

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
conv_index == 0 ? inputParameters.InputBlobName : modules.back()->GetOutputBlobNames()[0],
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
