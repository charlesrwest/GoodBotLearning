#include "RandomizedFileDataLoader.hpp"

#include<algorithm>
#include<cstring>
#include "SOMException.hpp"
#include<iostream>

using namespace GoodBot;

RandomizedFileDataLoader::RandomizedFileDataLoader(const std::string& sourceFilePath, int64_t inputBlobSize, int64_t outputBlobSize, int64_t numberOfBlobsPerBuffer, int64_t numberOfBuffers) : InputBlobSize(inputBlobSize), OutputBlobSize(outputBlobSize), NumberOfBlobsPerBuffer(numberOfBlobsPerBuffer), File(sourceFilePath, std::ifstream::binary | std::ifstream::ate), Randomness(RandomDevice())
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

//Intitialize buffers
RandomBlobIndexBuffers.resize(numberOfBuffers);
Buffers.resize(numberOfBuffers);

for(std::vector<char>& buffer : Buffers)
{
buffer.resize(numberOfBlobsPerBuffer*(inputBlobSize+outputBlobSize));
} 

FillBuffersIfEmpty();
}

bool RandomizedFileDataLoader::ReadBlob(char* inputBufferAddress, char* outputBufferAddress)
{
//We assume the buffers have at least one example in them
int64_t buffer_to_read_from = GetRandomNonEmptyBufferIndex();

std::queue<int64_t>& randomBlobIndices = RandomBlobIndexBuffers[buffer_to_read_from];

const std::vector<char>& buffer = Buffers[buffer_to_read_from];

//Read blobs from the buffer
int64_t blob_index = randomBlobIndices.front();
randomBlobIndices.pop();

std::memcpy(inputBufferAddress, &buffer[blob_index*(InputBlobSize+OutputBlobSize)], InputBlobSize);
std::memcpy(outputBufferAddress, &buffer[blob_index*(InputBlobSize+OutputBlobSize) + InputBlobSize], OutputBlobSize);

return FillBuffersIfEmpty();
}

int64_t RandomizedFileDataLoader::GetInputDataSize() const
{
return InputBlobSize;
}

int64_t RandomizedFileDataLoader::GetExpectedOutputDataSize() const
{
return OutputBlobSize;
}

int64_t RandomizedFileDataLoader::GetRandomNonEmptyBufferIndex()
{
    //Assemble list of non-empty buffers
    std::vector<int64_t> non_empty_buffers;

    for(int64_t buffer_index = 0; buffer_index < RandomBlobIndexBuffers.size(); buffer_index++)
    {
        if(RandomBlobIndexBuffers[buffer_index].size() != 0)
        {
            non_empty_buffers.emplace_back(buffer_index);
        }
    }

    SOM_ASSERT(non_empty_buffers.size() > 0, "There are no non-empty buffers");
    std::uniform_int_distribution<int64_t> buffer_distribution(0, non_empty_buffers.size()-1);
    int64_t non_empty_buffer_index = buffer_distribution(Randomness);

    int64_t buffer_to_use = non_empty_buffers[non_empty_buffer_index];
    SOM_ASSERT((buffer_to_use >= 0) && (buffer_to_use < RandomBlobIndexBuffers.size()), "Invalid buffer index about to be returned.");

    return non_empty_buffers[non_empty_buffer_index];
}

bool RandomizedFileDataLoader::FillBuffersIfEmpty()
{
    std::vector<int64_t> empty_buffers;
    for(int64_t buffer_index = 0; buffer_index < RandomBlobIndexBuffers.size(); buffer_index++)
    {
        if(RandomBlobIndexBuffers[buffer_index].size() == 0)
        {
            empty_buffers.emplace_back(buffer_index);
        }
    }

    bool epoc_finished = empty_buffers.size() == RandomBlobIndexBuffers.size();

    for(int64_t empty_buffer_index = 0; empty_buffer_index < empty_buffers.size(); empty_buffer_index++)
    {
        if( FileOffsets.size() == 0)
        {
            if(epoc_finished)
            {
                FillFileOffsets();
            }
            else
            {
                //We don't have any examples left in the file to load and we haven't used up all the ones in the buffers, so leave empty for now
                continue;
            }
        }

        FillBuffer(empty_buffers[empty_buffer_index]);
    }

    return epoc_finished;
}

void RandomizedFileDataLoader::FillBuffer(int64_t bufferIndex)
{
    int64_t next_file_offset = FileOffsets.front();
    FileOffsets.pop();

    std::queue<int64_t>& randomBlobIndices = RandomBlobIndexBuffers[bufferIndex];
    std::vector<char>& buffer = Buffers[bufferIndex];

    File.seekg(next_file_offset);
    File.read(((char *) &(buffer[0])), (InputBlobSize+OutputBlobSize)*NumberOfBlobsPerBuffer);

    //Add random indices to queue
    std::vector<int64_t> random_indices(NumberOfBlobsPerBuffer);
    std::iota(random_indices.begin(), random_indices.end(), 0);

    std::shuffle(random_indices.begin(), random_indices.end(), Randomness);

    //Add the randomized indices to the indices queue for this buffer
    if(!randomBlobIndices.empty())
    {
        throw SOM_EXCEPTION("randomBlobIndices should be empty");
    }

    for(int64_t random_index : random_indices)
    {
        randomBlobIndices.push(random_index);
    }
}

void RandomizedFileDataLoader::FillFileOffsets()
{
    int64_t buffer_byte_size = (InputBlobSize + OutputBlobSize)*NumberOfBlobsPerBuffer;

    std::vector<int64_t> file_offsets_buffer;
    for(int64_t file_offset = 0; file_offset < (FileSizeInBytes - buffer_byte_size); file_offset+=buffer_byte_size)
    {
        file_offsets_buffer.emplace_back(file_offset);
    }

    std::shuffle(file_offsets_buffer.begin(), file_offsets_buffer.end(), Randomness);

    for(int64_t file_offset : file_offsets_buffer)
    {
        FileOffsets.push(file_offset);
    }
}
