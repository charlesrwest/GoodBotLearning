#include "MemoryDataLoader.hpp"

#include "SOMException.hpp"

using namespace GoodBot;

MemoryDataLoader::MemoryDataLoader(int64_t numberOfExamples, const char* inputData, int64_t exampleInputSize,
                 const char* expectedOutputData, int64_t exampleExpectedOutputSize)
{
    Initialize(numberOfExamples, inputData, exampleInputSize, expectedOutputData, exampleExpectedOutputSize);
}

MemoryDataLoader::MemoryDataLoader(int64_t numberOfExamples, const std::vector<char>& inputData, const std::vector<char>& expectedOutputData)
{
    SOM_ASSERT((inputData.size() % numberOfExamples) == 0, "Input vector size does not divide by number of examples");
    SOM_ASSERT((expectedOutputData.size() % numberOfExamples) == 0, "Expected output vector size does not divide by number of examples");

    Initialize(numberOfExamples, &inputData[0], inputData.size() / numberOfExamples, &expectedOutputData[0], expectedOutputData.size() / numberOfExamples);
}

bool MemoryDataLoader::ReadBlob(char* inputBufferAddress, char* expectedOutputBufferAddress)
{
    int64_t current_example_index = blob_indices[blob_index_index];
    memcpy(inputBufferAddress, (&input_data[0]) + current_example_index*size_of_input, size_of_input);
    memcpy(expectedOutputBufferAddress, (&expected_output_data[0]) + current_example_index*size_of_output, size_of_output);

    blob_index_index++;
    if(blob_index_index == blob_indices.size())
    {
        ResetExampleIndexIndexAndShuffleIndices();
        return true;
    }

    return false;
}

int64_t MemoryDataLoader::GetInputDataSize() const
{
    return size_of_input;
}

int64_t MemoryDataLoader::GetExpectedOutputDataSize() const
{
    return size_of_output;
}

int64_t MemoryDataLoader::GetNumberOfExamples() const
{
    return input_data.size() / GetInputDataSize();
}

void MemoryDataLoader::Initialize(int64_t numberOfExamples, const char* inputData, int64_t exampleInputSize,
                 const char* expectedOutputData, int64_t exampleExpectedOutputSize)
{
SOM_ASSERT(numberOfExamples > 0, "No data in loader");
SOM_ASSERT(inputData != nullptr, "Invalid input pointer");
SOM_ASSERT(exampleInputSize > 0, "Zero or negative example input size?");
SOM_ASSERT(expectedOutputData != nullptr, "Invalid expectedOutputData pointer");
SOM_ASSERT(exampleExpectedOutputSize > 0, "Zero or negative expected output size?");

size_of_input = exampleInputSize;
size_of_output = exampleExpectedOutputSize;

input_data.resize(numberOfExamples*size_of_input);
expected_output_data.resize(numberOfExamples*size_of_output);

memcpy((&input_data[0]), inputData, numberOfExamples*size_of_input);
memcpy((&expected_output_data[0]), expectedOutputData, numberOfExamples*size_of_output);

for(int64_t example_index = 0; example_index < numberOfExamples; example_index++)
{
    blob_indices.emplace_back(example_index);
}

ResetExampleIndexIndexAndShuffleIndices();
}


void MemoryDataLoader::ResetExampleIndexIndexAndShuffleIndices()
{
    blob_index_index = 0;

    std::shuffle(blob_indices.begin(), blob_indices.end(), mersenne);
}
