#define CATCH_CONFIG_MAIN //Make main function automatically
#include "catch.hpp"
#include<cstdlib>
#include<string>

#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include<google/protobuf/text_format.h>
#include "FullyConnectedModuleDefinition.hpp"
#include<iostream>
#include<cmath>
#include<cassert>
#include "AveragedL2LossModuleDefinition.hpp"
#include "AdamSolver.hpp"
#include<fstream>
#include<random>
#include<cstdio>
#include "DataLoader.hpp"
#include "SOMScopeGuard.hpp"

const double PI  =3.141592653589793238463;
const float  PI_F=3.14159265358979f;

template<class DataType>
void PairedRandomShuffle(typename std::vector<DataType>& inputData, typename  std::vector<DataType>& expectedOutputData)
{
assert(inputData.size() == expectedOutputData.size());

//Fisher-Yates shuffle
for(typename std::vector<DataType>::size_type index = 0; index < inputData.size(); index++)
{
typename std::vector<DataType>::size_type elementToSwapWithIndex = index + (rand() % (inputData.size() - index));
std::swap(inputData[index], inputData[elementToSwapWithIndex]);
std::swap(expectedOutputData[index], expectedOutputData[elementToSwapWithIndex]);
}
};


/*
//Not truly a test case, example of training using the new compute module way of specifying networks.
TEST_CASE("Test generated Fully Connected NetDefs", "[Example]")
{
//Make 10000 training examples
int64_t numberOfTrainingExamples = 10000;
int64_t batch_size = 2;
REQUIRE((numberOfTrainingExamples % batch_size) == 0);
std::vector<float> trainingInputs;
std::vector<float> trainingExpectedOutputs;

for(int64_t trainingExampleIndex = 0; trainingExampleIndex < numberOfTrainingExamples; trainingExampleIndex++)
{
trainingInputs.emplace_back((((double) trainingExampleIndex)/(numberOfTrainingExamples+1))*2.0 - 1.0);
trainingExpectedOutputs.emplace_back(sin(((double) trainingExampleIndex)/(numberOfTrainingExamples+1)*2.0*PI)*.5 + .5);
}

//Shuffle the examples
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);

//Create the Caffe2 workspace/context
caffe2::Workspace workspace;
caffe2::CPUContext context;

//Create the blobs to inject the sine input/output for training
caffe2::TensorCPU& inputBlob = *workspace.CreateBlob("inputBlob")->GetMutable<caffe2::TensorCPU>();
inputBlob.Resize(batch_size, 1);
inputBlob.mutable_data<float>();

std::string trainingExpectedOutputBlobName = "trainingExpectedOutputBlobName";
caffe2::TensorCPU& expectedOutputBlob = *workspace.CreateBlob(trainingExpectedOutputBlobName)->GetMutable<caffe2::TensorCPU>();
expectedOutputBlob.Resize(batch_size, 1);
expectedOutputBlob.mutable_data<float>();

//Define a 3 layer fully connected network with default (sigmoidal) activation
GoodBot::FullyConnectedModuleDefinition network("inputBlob", {100, 100, 1}, "HelloNetwork", 1);
network.SetMode("TRAIN");

//Add a module for computing loss to the end of the network
GoodBot::AveragedL2LossModuleDefinitionParameters lossParameters;
lossParameters.inputBlobName = network.GetOutputBlobNames()[0];
lossParameters.moduleName = "AveragedL2Loss";
lossParameters.trainingExpectedOutputBlobName = trainingExpectedOutputBlobName;
lossParameters.testExpectedOutputBlobName = trainingExpectedOutputBlobName;

network.AddModule(*(new  GoodBot::AveragedL2LossModuleDefinition(lossParameters)));
network.SetMode("TRAIN");

//Add a solver module for training/updating
GoodBot::AdamSolverParameters solverParams;
solverParams.moduleName = "AdamSolver";
solverParams.trainableParameterNames = network.GetTrainableBlobNames();
solverParams.trainableParameterShapes = network.GetTrainableBlobShapes();

network.AddModule(*(new GoodBot::AdamSolver(solverParams)));


SECTION("Test training network", "[networkArchitecture]")
{
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
print(trainingNetworkInitializationDefinition);

//Create and run the initialization network.
caffe2::NetBase* initializationNetwork = workspace.CreateNet(trainingNetworkInitializationDefinition);
initializationNetwork->Run();

//Automatically generate the training network
caffe2::NetDef trainingNetworkDefinition = network.GetNetwork(workspace.Blobs());

print(trainingNetworkDefinition);

//Instance the training network implementation
caffe2::NetBase* trainingNetwork = workspace.CreateNet(trainingNetworkDefinition);


//Create the deploy version of the network
network.SetMode("DEPLOY");

caffe2::NetDef deployNetworkDefinition = network.GetNetwork(workspace.Blobs());

caffe2::NetBase* deployNetwork = workspace.CreateNet(deployNetworkDefinition);

//Get the blob for network output/iteration count for later testing
caffe2::TensorCPU& networkOutput = *workspace.GetBlob(network.GetOutputBlobNames()[0])->GetMutable<caffe2::TensorCPU>();

caffe2::TensorCPU& iter = *workspace.GetBlob("AdamSolver_iteration_iterator")->GetMutable<caffe2::TensorCPU>();

//Train the network
int64_t numberOfTrainingIterations = 100000;

for(int64_t iteration = 0; iteration < numberOfTrainingIterations; iteration++)
{
//Shuffle data every epoc through
if((iteration % trainingInputs.size()) == 0)
{
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);
}

//Load data into blobs
memcpy(inputBlob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()], inputBlob.nbytes());
memcpy(expectedOutputBlob.mutable_data<float>(), &trainingExpectedOutputs[iteration % trainingExpectedOutputs.size()], expectedOutputBlob.nbytes());

//Run network with loaded instance
trainingNetwork->Run();

}

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

}


}
*/

TEST_CASE("Run iterator network and get data", "[Example]")
{
caffe2::Workspace workspace;
caffe2::CPUContext context;

std::string IteratorBlobName = "IterBlob";
caffe2::OperatorDef iterInitializationOperator;

iterInitializationOperator.set_type("ConstantFill");
iterInitializationOperator.add_output(IteratorBlobName);
caffe2::Argument& shape = *iterInitializationOperator.add_arg();
shape.set_name("shape");
shape.add_ints(1);
caffe2::Argument& value = *iterInitializationOperator.add_arg();
value.set_name("value");
value.set_i(0);
caffe2::Argument& dataType = *iterInitializationOperator.add_arg();
dataType.set_name("dtype");
dataType.set_i(caffe2::TensorProto_DataType_INT64); 

caffe2::NetDef initNetwork;
initNetwork.set_name("iter_init");

*initNetwork.add_op() = iterInitializationOperator;

caffe2::NetBase* initializationNetwork = workspace.CreateNet(initNetwork);
initializationNetwork->Run();

caffe2::OperatorDef iterOperator;

iterOperator.set_type("Iter");
iterOperator.add_input(IteratorBlobName);
iterOperator.add_output(IteratorBlobName);

caffe2::NetDef iterNetwork;
iterNetwork.set_name("iter_net");

*iterNetwork.add_op() = iterOperator;

caffe2::NetBase* iterationNetwork = workspace.CreateNet(iterNetwork);

caffe2::TensorCPU& iteratorBlob = *workspace.GetBlob(IteratorBlobName)->GetMutable<caffe2::TensorCPU>();

for(int64_t counter = 0; counter < 10000; counter++)
{
REQUIRE(iteratorBlob.mutable_data<int64_t>()[0] == counter);

iterationNetwork->Run();
}

}

//Returns sum of outputted sequence (can wrap around)
uint64_t WriteOutSequentialNumbers(uint64_t startValue, int64_t numberOfEntries, std::ofstream& outputStream)
{
uint64_t sum = 0;
for(int64_t entry_index = 0; entry_index < numberOfEntries; entry_index++)
{
uint64_t buffer = startValue + entry_index;
sum = sum + buffer;
outputStream.write((char*) &buffer, sizeof(uint64_t));
}

return sum;
} 

TEST_CASE("Write blobber file and pseudo-randomly read from it", "[Example]")
{
//Inputs are N sequential number starting at a random place and outputs are sequences of integers which start at the sum of the input sequence.
std::random_device randomness;
std::uniform_int_distribution<uint64_t> random_integer_generator(0, std::numeric_limits<uint64_t>::max());


auto TryCombination = [&](int64_t numberOfEntries, int64_t inputSequenceLength, int64_t outputSequenceLength, int64_t bufferSize, int64_t numberOfBuffers, int64_t maxRereadsBeforeRefill, int64_t numberOfTestReads, int64_t batchSize)
{
std::string temp_file_name = "temp.blobber";

//Delete the temporary file when we leave this function
SOMScopeGuard file_deleter([&](){ remove(temp_file_name.c_str()); });

int64_t input_blob_size = inputSequenceLength*sizeof(uint64_t);
int64_t output_blob_size = outputSequenceLength*sizeof(uint64_t);

//Write out the file
{
std::ofstream output_stream(temp_file_name,  std::ofstream::binary);

for(int64_t entry_index = 0; entry_index < numberOfEntries; entry_index++)
{
uint64_t output_sequence_start = 0;
uint64_t input_sequence_start = random_integer_generator(randomness);

//Write input sequence followed by output sequence
output_sequence_start = WriteOutSequentialNumbers(input_sequence_start, inputSequenceLength, output_stream);

WriteOutSequentialNumbers(output_sequence_start, outputSequenceLength, output_stream);
}
}

//Check file size
{
std::ifstream input_stream(temp_file_name,  std::ifstream::binary);

std::streampos start_position = input_stream.tellg();

input_stream.seekg(0, std::ifstream::end);

int64_t file_size = input_stream.tellg() - start_position;

REQUIRE(file_size == ((input_blob_size+output_blob_size)*numberOfEntries));
}

//Make a data loader and see if the retrieved blobs are coherent (match input/output rule)
GoodBot::DataLoader loader(temp_file_name, input_blob_size, output_blob_size, bufferSize, numberOfBuffers, maxRereadsBeforeRefill);

std::vector<uint64_t> input_buffer(inputSequenceLength*batchSize);
std::vector<uint64_t> output_buffer(outputSequenceLength*batchSize);

for(int64_t test_read_index = 0; test_read_index < numberOfTestReads; test_read_index++)
{
loader.ReadBlobs((char*) &input_buffer[0], (char*) &output_buffer[0], batchSize);

//Check each blob in batch
for(int64_t blob_index = 0; blob_index < batchSize; blob_index++)
{
int64_t input_sequence_offset = blob_index*inputSequenceLength;
uint64_t input_sum = input_buffer[input_sequence_offset];

//Check that input is a sequence
for(int64_t sequence_index = 1; sequence_index < inputSequenceLength; sequence_index++)
{
REQUIRE((input_buffer[sequence_index - 1 + input_sequence_offset]+1) == input_buffer[sequence_index + input_sequence_offset]);
input_sum += input_buffer[sequence_index + input_sequence_offset];
}

//Make sure output start matches
int64_t output_sequence_offset = blob_index*outputSequenceLength;
REQUIRE(output_buffer[output_sequence_offset] == input_sum);

//Check that output is a sequence
for(int64_t sequence_index = 1; sequence_index < outputSequenceLength; sequence_index++)
{
REQUIRE((output_buffer[sequence_index - 1 + output_sequence_offset]+1) == output_buffer[sequence_index + output_sequence_offset]);
}
}

}

};
 
//Write/read files with 1000 entries with input sequences of length 10 and output sequences of length 3, 5 buffers of 100 entries, with a max 2 rereads and 1000000 test reads, batch size 3
TryCombination(1000, 10, 3, 100, 5, 2, 1000000, 3);
}

