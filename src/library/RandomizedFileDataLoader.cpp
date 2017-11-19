#include "RandomizedFileDataLoader.hpp"

#include<algorithm>
#include<cstring>
#include "SOMException.hpp"
#include<iostream>

using namespace GoodBot;

RandomizedFileDataLoader::RandomizedFileDataLoader(const std::string& sourceFilePath, int64_t inputBlobSize, int64_t outputBlobSize, int64_t numberOfBlobsPerBuffer, int64_t numberOfBuffers, int64_t maxRereadsBeforeRefill) : InputBlobSize(inputBlobSize), OutputBlobSize(outputBlobSize), NumberOfBlobsPerBuffer(numberOfBlobsPerBuffer), File(sourceFilePath, std::ifstream::binary | std::ifstream::ate), MaxRereadsBeforeRefill(maxRereadsBeforeRefill), BufferSelector(0, numberOfBuffers-1), Randomness(RandomDevice())
{
//Check file size (we opened at the end)
FileSizeInBytes = File.tellg();

if((FileSizeInBytes % (inputBlobSize + outputBlobSize)) != 0)
{
throw SOM_EXCEPTION("Data file is incorrect size");
}

int64_t number_of_examples_in_file = FileSizeInBytes/(inputBlobSize + outputBlobSize);

if(number_of_examples_in_file < numberOfBlobsPerBuffer)
{
throw SOM_EXCEPTION("Data file is too small, there are less examples (" + std::to_string(number_of_examples_in_file) + ") then a single buffer " + std::to_string(numberOfBlobsPerBuffer));
}

FileBlobIndexSelector = std::uniform_int_distribution<int64_t>(0, number_of_examples_in_file - numberOfBlobsPerBuffer);

//Intitialize buffers
RandomBlobIndexBuffers.resize(numberOfBuffers);
InputBuffers.resize(numberOfBuffers);
OutputBuffers.resize(numberOfBuffers);

for(std::vector<char>& input_buffer : InputBuffers)
{
input_buffer.resize(numberOfBlobsPerBuffer*inputBlobSize);
} 

for(std::vector<char>& output_buffer : OutputBuffers)
{
output_buffer.resize(numberOfBlobsPerBuffer*outputBlobSize);
} 
}

bool RandomizedFileDataLoader::ReadBlob(char* inputBufferAddress, char* outputBufferAddress)
{
//Choose a buffer
int64_t buffer_index = BufferSelector(Randomness);

std::queue<int64_t>& randomBlobIndices = RandomBlobIndexBuffers[buffer_index];

const std::vector<char>& input_buffer = InputBuffers[buffer_index];
const std::vector<char>& output_buffer = OutputBuffers[buffer_index];

//If that buffer needs to be refilled, fill it
if(randomBlobIndices.size() == 0)
{
FillBuffer(buffer_index);
}

//Read blobs from the buffers
int64_t blob_index = randomBlobIndices.front();
randomBlobIndices.pop();

std::memcpy(inputBufferAddress, &input_buffer[blob_index*InputBlobSize], InputBlobSize);
std::memcpy(outputBufferAddress, &output_buffer[blob_index*OutputBlobSize], OutputBlobSize);

return false;
}

int64_t RandomizedFileDataLoader::GetInputDataSize() const
{
return InputBlobSize;
}

int64_t RandomizedFileDataLoader::GetExpectedOutputDataSize() const
{
return OutputBlobSize;
}


void RandomizedFileDataLoader::FillBuffer(int64_t bufferIndex)
{
std::queue<int64_t>& randomBlobIndices = RandomBlobIndexBuffers[bufferIndex];

const std::vector<char>& input_buffer = InputBuffers[bufferIndex];
const std::vector<char>& output_buffer = OutputBuffers[bufferIndex];

//Pick a random example in the file to start the buffer at
int64_t example_read_start_index = FileBlobIndexSelector(Randomness);

File.seekg(example_read_start_index*(InputBlobSize + OutputBlobSize));

for(int64_t example_index = 0; example_index < NumberOfBlobsPerBuffer; example_index++)
{
File.read(((char *) &(input_buffer[0])) + example_index*InputBlobSize, InputBlobSize);
File.read(((char *) &(output_buffer[0])) + example_index*OutputBlobSize, OutputBlobSize);
}

//Add random indices to queue
std::vector<int64_t> sequential_buffer_indices(NumberOfBlobsPerBuffer);
std::iota(sequential_buffer_indices.begin(), sequential_buffer_indices.end(), 0);

//Fill list with N sequential copies of the possible indice list and then randomize
std::vector<int64_t> random_order_buffer_indices;
random_order_buffer_indices.reserve(NumberOfBlobsPerBuffer*MaxRereadsBeforeRefill);
for(int64_t read_index = 0; read_index < MaxRereadsBeforeRefill; read_index++)
{
random_order_buffer_indices.insert(random_order_buffer_indices.end(), sequential_buffer_indices.begin(), sequential_buffer_indices.end());
}

std::shuffle(random_order_buffer_indices.begin(), random_order_buffer_indices.end(), Randomness);

//Add the randomized indices to the indices queue for this buffer
if(!randomBlobIndices.empty())
{
throw SOM_EXCEPTION("randomBlobIndices should be empty");
}

for(int64_t random_index : random_order_buffer_indices)
{
randomBlobIndices.push(random_index);
}
}
