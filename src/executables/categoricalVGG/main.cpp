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


int main(int argc, char **argv)
{
//Create the Caffe2 workspace/context
caffe2::Workspace workspace;
caffe2::CPUContext context;


//Create the blobs to inject the sine input/output for training
caffe2::TensorCPU& inputBlob = *workspace.CreateBlob("inputBlob")->GetMutable<caffe2::TensorCPU>();
inputBlob.Resize(1, 3, 256, 256);
inputBlob.mutable_data<uint8_t>();

std::string trainingExpectedOutputBlobName = "expectedOutputBlobName";
caffe2::TensorCPU& expectedOutputBlob = *workspace.CreateBlob(trainingExpectedOutputBlobName)->GetMutable<caffe2::TensorCPU>();
expectedOutputBlob.Resize(1, 1);
expectedOutputBlob.mutable_data<int32_t>();

//Define a VGG16 network
GoodBot::VGG16Parameters VGG_param;

VGG_param.Name = "0";
VGG_param.InputBlobName = "inputBlob";
VGG_param.OutputBlobName = "outputBlob";
VGG_param.TrainingExpectedOutputBlobName = "expectedOutputBlobName";
VGG_param.TestExpectedOutputBlobName = "expectedOutputBlobName";
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
std::cout << "Network has " << network.GetOutputBlobNames().size() << std::endl << std::flush;

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

//Print out the generated network architecture
//print(trainingNetworkInitializationDefinition);

//Create and run the initialization network.
caffe2::NetBase* initializationNetwork = workspace.CreateNet(trainingNetworkInitializationDefinition);

std::cout << "About to initialize network" << std::endl << std::flush;
//initializationNetwork->Run();
std::cout << "Network initialized" << std::endl << std::flush;

//Automatically generate the training network
//caffe2::NetDef trainingNetworkDefinition = network.GetNetwork(workspace.Blobs());

std::cout << "Created training network" << std::endl << std::flush;

//print(trainingNetworkDefinition);

//Instance the training network implementation
//caffe2::NetBase* trainingNetwork = workspace.CreateNet(trainingNetworkDefinition);

//Setup IO with the training data set
GoodBot::DataLoader loader("../data/trainingData.blobber", 3*256*256*sizeof(uint8_t), sizeof(int32_t), 5, 2, 1);

//Train the network
int64_t numberOfTrainingIterations = 10000000;

for(int64_t iteration = 0; iteration < numberOfTrainingIterations; iteration++)
{
//Load data into blobs
loader.ReadBlobs((char *) inputBlob.mutable_data<uint8_t>(), (char *) expectedOutputBlob.mutable_data<int32_t>(), 1);

std::cout << "Training network" << std::endl << std::flush;

//Run network with loaded instance
//trainingNetwork->Run();
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
memcpy(inputBlob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()], inputBlob.nbytes());

deployNetwork->Run();

pretrainedDeployResults << *inputBlob.mutable_data<float>() << ", " << *networkOutput.mutable_data<float>() << ", " <<  *iter.mutable_data<int64_t>() << std::endl;
}
}
*/

return 0;
} 
