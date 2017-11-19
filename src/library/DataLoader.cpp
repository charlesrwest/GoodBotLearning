#include "DataLoader.hpp"

using namespace GoodBot;

bool DataLoader::ReadBlobs(char* inputBufferAddress, char* expectedOutputBufferAddress, int64_t numberOfBlobs)
{
    bool hit_epoc_end = false;
    for(int64_t example_index = 0; example_index < numberOfBlobs; example_index++)
    {
        hit_epoc_end |= ReadBlob(inputBufferAddress + example_index*GetInputDataSize(), expectedOutputBufferAddress + example_index*GetExpectedOutputDataSize());
    }

    return hit_epoc_end;
}
