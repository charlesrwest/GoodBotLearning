#include "VGGConvolutionModule.hpp"
#include "ConvModule.hpp"
#include "RELUModule.hpp"
#include<iostream>

using namespace GoodBot;

VGGConvolutionModule::VGGConvolutionModule(const VGGConvolutionModuleParameters& inputParameters)
{
SetName(inputParameters.Name);

GoodBot::ConvModuleParameters conv_module_params;
conv_module_params.InputBlobName = inputParameters.InputBlobName;
conv_module_params.Stride = 1;
conv_module_params.PaddingSize = 1;
conv_module_params.KernelSize = 3;

int64_t previous_layer_depth = inputParameters.InputDepth;
for(int64_t moduleIndex = 0; moduleIndex < inputParameters.NumberOfConvolutions; moduleIndex++)
{
conv_module_params.OutputBlobName = Name() + "_" + std::to_string(moduleIndex);
conv_module_params.Name = conv_module_params.OutputBlobName;

if(moduleIndex != 0)
{
conv_module_params.InputBlobName = Name() + "_" + std::to_string(moduleIndex-1);
}

conv_module_params.Name = Name() + "_conv" + std::to_string(moduleIndex);
conv_module_params.InputDepth = previous_layer_depth;
conv_module_params.OutputDepth = inputParameters.OutputDepth;

AddModule(*(new ConvModule(conv_module_params)));
previous_layer_depth = inputParameters.OutputDepth;
}
LastConvLayerIndex = modules.size() - 1;

AddModule(*(new RELUModule({Name() + "_relu", modules.back()->GetOutputBlobNames()[0], modules.back()->GetOutputBlobNames()[0]})));
}

std::vector<std::string> VGGConvolutionModule::GetInputBlobNames() const
{
return modules.front()->GetInputBlobNames();
}

std::vector<std::string> VGGConvolutionModule::GetOutputBlobNames() const
{
return modules[LastConvLayerIndex+1]->GetOutputBlobNames();
}
