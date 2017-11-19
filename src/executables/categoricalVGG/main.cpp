#include<cstdlib>
#include<string>

#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include<google/protobuf/text_format.h>
#include "VGG16.hpp"
#include<iostream>
#include<cmath>
#include<cassert>
#include "AveragedL2LossModuleDefinition.hpp"
#include "AdamSolver.hpp"
#include "SOMException.hpp"
#include "caffe2/core/init.h"
#include "caffe2/core/context_gpu.h"
#include "RandomizedFileDataLoader.hpp"
#include "DataSynchronizer.hpp"
#include "AveragedLossLayerDefinition.hpp"
#include "LabelCrossEntropyOperator.hpp"
#include<chrono>


int main(int argc, char **argv)
{
//Create the Caffe2 workspace/context
caffe2::DeviceOption option;
option.set_device_type(caffe2::CUDA);
caffe2::CUDAContext cuda_context(option);

//caffe2::CPUContext cpu_context;

caffe2::Workspace workspace;


//Create the blobs to inject the sine input/output for training
caffe2::TensorCPU& input_blob_cpu = *workspace.CreateBlob("input_blob_cpu")->GetMutable<caffe2::TensorCPU>();
input_blob_cpu.Resize(1, 3, 224, 224);
input_blob_cpu.mutable_data<uint8_t>();

//std::array<uint8_t, 3*224*224> cpu_input_buffer;
//std::array<uint32_t, 1> cpu_expected_output_buffer;

std::string training_expected_output_blob_name = "expected_output_blob_cpu";
caffe2::TensorCPU& expected_output_blob_cpu = *workspace.CreateBlob(training_expected_output_blob_name)->GetMutable<caffe2::TensorCPU>();
expected_output_blob_cpu.Resize(1, 1);
expected_output_blob_cpu.mutable_data<int32_t>();

//Define a GPU version of the blobs and a network to move data back and forth between them
caffe2::TensorCUDA& input_blob_gpu = *workspace.CreateBlob("input_blob_gpu")->GetMutable<caffe2::TensorCUDA>();
input_blob_gpu.Resize(1, 3, 224, 224);
input_blob_gpu.mutable_data<uint8_t>();

std::string training_expected_output_blob_name_gpu = "expected_output_blob_gpu";
caffe2::TensorCUDA& expected_output_blob_gpu = *workspace.CreateBlob(training_expected_output_blob_name_gpu)->GetMutable<caffe2::TensorCUDA>();
expected_output_blob_gpu.Resize(1, 1);
expected_output_blob_gpu.mutable_data<int32_t>();

//Define a VGG16 network
GoodBot::VGG16Parameters VGG_param;

VGG_param.Name = "0";
VGG_param.InputBlobName = "input_blob_gpu";
VGG_param.OutputBlobName = "outputBlob";
VGG_param.TrainingExpectedOutputBlobName = training_expected_output_blob_name_gpu;
VGG_param.TestExpectedOutputBlobName = training_expected_output_blob_name_gpu;
VGG_param.BatchSize = 1;

GoodBot::VGG16 network(VGG_param);

//VGG outputs softmax
//Training adds LabelCrossEntropy, then averaged loss

network.AddModule(*(new GoodBot::LabelCrossEntropyOperator(network.Name() + "_label_cross_entropy", network.GetOutputBlobNames()[0], training_expected_output_blob_name_gpu)));

network.AddModule(*(new GoodBot::AveragedLossLayerDefinition({network.Name() + "_label_cross_entropy", network.Name() + "_averaged_loss"})));


network.SetMode("TRAIN");

//refactor vgg16 to output softmax and then make a composite module for averaged loss which is passthrough unless in training mode

/*
std::cout << "Network modules: " << std::endl << std::flush;

std::vector<std::string> module_names = network.GetModuleNames();

for(const std::string& module_name : module_names)
{
std::cout << module_name << std::endl << std::flush;
}
*/

//Get the name of the loss blob
//std::cout << "Network has " << network.GetOutputBlobNames().size() << std::endl << std::flush;

//Add a solver module for training/updating
GoodBot::AdamSolverParameters solverParams;
solverParams.moduleName = network.Name() + "adam_solver";
solverParams.trainableParameterNames = network.GetTrainableBlobNames();
solverParams.trainableParameterShapes = network.GetTrainableBlobShapes();

network.AddModule(*(new GoodBot::AdamSolver(solverParams)));

//Add function to allow printing of network architectures
std::function<void(const google::protobuf::Message&)> print = [&](const google::protobuf::Message& inputMessage)
{
std::string buffer;

google::protobuf::TextFormat::PrintToString(inputMessage, &buffer);

std::cout << buffer<<std::endl;
};

//Training the network, so set the mode to train
network.SetMode("TRAIN");

//Initialize the network by automatically generating the NetDef for network initialization in "TRAIN" mode
caffe2::NetDef trainingNetworkInitializationDefinition = network.GetInitializationNetwork();
trainingNetworkInitializationDefinition.mutable_device_option()->set_device_type(caffe2::CUDA);

//Print out the generated network architecture
//print(trainingNetworkInitializationDefinition);

//Create and run the initialization network.
caffe2::NetBase* initializationNetwork = workspace.CreateNet(trainingNetworkInitializationDefinition);

std::cout << "About to initialize network" << std::endl << std::flush;
initializationNetwork->Run();
std::cout << "Network initialized" << std::endl << std::flush;

//Automatically generate the training network
caffe2::NetDef trainingNetworkDefinition = network.GetNetwork(workspace.Blobs());
trainingNetworkDefinition.mutable_device_option()->set_device_type(caffe2::CUDA);

std::cout << "Created training network" << std::endl << std::flush;

print(trainingNetworkDefinition);

//Instance the training network implementation
caffe2::NetBase* trainingNetwork = workspace.CreateNet(trainingNetworkDefinition);

//workspace.PrintBlobSizes();

//Setup IO with the training data set
GoodBot::RandomizedFileDataLoader loader("../data/trainingData.blobber", 3*224*224*sizeof(uint8_t), sizeof(int32_t), 10, 1, 1);

GoodBot::DataSynchronizer training_data_synchronizer("training_data_synchronizer", {{"input_blob_cpu", "input_blob_gpu"}, {"expected_output_blob_cpu", "expected_output_blob_gpu"}}, workspace);

//Run network once so outputs are initialized
loader.ReadBlobs((char *) input_blob_cpu.mutable_data<uint8_t>(), (char *) expected_output_blob_cpu.mutable_data<int32_t>(), 1);
training_data_synchronizer.MoveCPUDataToGPU();
trainingNetwork->Run();


//Make blobs/data synchronizer to retrieve output and loss 
std::string soft_max_cpu_name = "0_soft_max_cpu";
caffe2::TensorCPU& soft_max_cpu = *workspace.CreateBlob(soft_max_cpu_name)->GetMutable<caffe2::TensorCPU>();
soft_max_cpu.Resize(1, 2);
soft_max_cpu.mutable_data<float>();


std::string averaged_loss_cpu_name = "0_averaged_loss_cpu";
caffe2::TensorCPU& averaged_loss_cpu = *workspace.CreateBlob(averaged_loss_cpu_name)->GetMutable<caffe2::TensorCPU>();
averaged_loss_cpu.Resize(1, 1);
averaged_loss_cpu.mutable_data<float>();

GoodBot::DataSynchronizer output_synchronizer("output_synchronizer", {{soft_max_cpu_name, "0_soft_max"}, {averaged_loss_cpu_name, "0_averaged_loss"}}, workspace);

//workspace.PrintBlobSizes();

//Train the network
int64_t numberOfTrainingIterations = 10000000;

double exponential_moving_average = 0.0;
double exponential_moving_average_divisor = 1000.0;

std::ofstream log_file("log.csv");

for(int64_t iteration = 0; iteration < numberOfTrainingIterations; iteration++)
{
//Load data into blobs
loader.ReadBlobs((char *) input_blob_cpu.mutable_data<uint8_t>(), (char *) expected_output_blob_cpu.mutable_data<int32_t>(), 1);

training_data_synchronizer.MoveCPUDataToGPU();

//Run network with loaded instance
trainingNetwork->Run();

output_synchronizer.MoveGPUDataToCPU();

auto now = std::chrono::system_clock::now();
auto now_c = std::chrono::system_clock::to_time_t(now);

exponential_moving_average = exponential_moving_average*((exponential_moving_average_divisor-1.0)/exponential_moving_average_divisor) + averaged_loss_cpu.mutable_data<float>()[0]/exponential_moving_average_divisor;

if((iteration % 100) == 0)
{
log_file << (*expected_output_blob_cpu.mutable_data<int32_t>()) << ", " << (soft_max_cpu.mutable_data<float>()[0]) << ", " << (soft_max_cpu.mutable_data<float>()[1]) << ", " << (averaged_loss_cpu.mutable_data<float>()[0]) << ", " << exponential_moving_average << std::endl;

log_file << std::flush;
}

//std::cout << "Hello" << std::endl << std::flush;

//std::cout << "Training network iter " << iteration << " loss " << (*loss.mutable_data<float>()) << std::endl << std::flush;
}

return 0;
} 
