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
#include "DataLoader.hpp"
#include "DataSynchronizer.hpp"


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

network.SetMode("TRAIN");

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
solverParams.moduleName = "AdamSolver";
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
GoodBot::DataLoader loader("../data/trainingData.blobber", 3*224*224*sizeof(uint8_t), sizeof(int32_t), 100, 10, 1);

GoodBot::DataSynchronizer training_data_synchronizer("training_data_synchronizer", {{"input_blob_cpu", "input_blob_gpu"}, {"expected_output_blob_cpu", "expected_output_blob_gpu"}}, workspace);

//Run network once so outputs are initialized
loader.ReadBlobs((char *) input_blob_cpu.mutable_data<uint8_t>(), (char *) expected_output_blob_cpu.mutable_data<int32_t>(), 1);
training_data_synchronizer.MoveCPUDataToGPU();
trainingNetwork->Run();

//Make blobs/data synchronizer to retrieve output and loss 
std::string soft_max_cpu_name = "0_soft_max_soft_max_cpu";
caffe2::TensorCPU& soft_max_cpu = *workspace.CreateBlob(soft_max_cpu_name)->GetMutable<caffe2::TensorCPU>();
expected_output_blob_cpu.Resize(1, 1);
expected_output_blob_cpu.mutable_data<int32_t>();

GoodBot::DataSynchronizer output_synchronizer("output_synchronizer", {{soft_max_cpu_name, "0_soft_max_soft_max"}}, workspace);



//Train the network
int64_t numberOfTrainingIterations = 10000000;

//caffe2::TensorCPU& loss = *workspace.GetBlob("0_soft_max_training_loss")->GetMutable<caffe2::TensorCPU>();

for(int64_t iteration = 0; iteration < numberOfTrainingIterations; iteration++)
{
//Load data into blobs
loader.ReadBlobs((char *) input_blob_cpu.mutable_data<uint8_t>(), (char *) expected_output_blob_cpu.mutable_data<int32_t>(), 1);

training_data_synchronizer.MoveCPUDataToGPU();

//Run network with loaded instance
trainingNetwork->Run();

output_synchronizer.MoveGPUDataToCPU();

std::cout << "Chugga: Expected " << (*expected_output_blob_cpu.mutable_data<int32_t>()) << " Output " << (*soft_max_cpu.mutable_data<int32_t>()) <<  std::endl << std::flush;

//std::cout << "Training network iter " << iteration << " loss " << (*loss.mutable_data<float>()) << std::endl << std::flush;
}

/*
//Create the deploy version of the network
network.SetMode("DEPLOY");

caffe2::NetDef deployNetworkDefinition = network.GetNetwork(workspace.Blobs());

caffe2::NetBase* deployNetwork = workspace.CreateNet(deployNetworkDefinition);

//Get the blob for network output/iteration count for later testing
caffe2::TensorCPU& networkOutput = *workspace.GetBlob(network.GetOutputBlobNames()[0])->GetMutable<caffe2::TensorCPU>();

caffe2::TensorCPU& iter = *workspace.GetBlob("AdamSolver_iteration_iterator")->GetMutable<caffe2::TensorCPU>();

//Output deploy results
{
std::ofstream pretrainedDeployResults("postTrainingDeployResults.csv");
for(int64_t iteration = 0; iteration < trainingInputs.size(); iteration++)
{
//Load data into blobs to csv for viewing
memcpy(input_blob_cpu.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()], input_blob_cpu.nbytes());

deployNetwork->Run();

pretrainedDeployResults << *input_blob_cpu.mutable_data<float>() << ", " << *networkOutput.mutable_data<float>() << ", " <<  *iter.mutable_data<int64_t>() << std::endl;
}
}
*/

return 0;
} 
