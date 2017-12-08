#include "SequentialFileDataLoader.hpp"
#include "SOMException.hpp"
#include<cstring>

using namespace GoodBot;

SequentialFileDataLoader::SequentialFileDataLoader(const std::string& sourceFilePath, int64_t inputBlobSize, int64_t outputBlobSize, int64_t numberOfBlobsInBuffer)
    : InputBlobSize(inputBlobSize), OutputBlobSize(outputBlobSize), NumberOfBlobsInBuffer(numberOfBlobsInBuffer), File(sourceFilePath, std::ifstream::binary | std::ifstream::ate)
{
    //Check file size (we opened at the end)
    FileSizeInBytes = File.tellg();

    if((FileSizeInBytes % (inputBlobSize + outputBlobSize)) != 0)
    {
        throw SOM_EXCEPTION("Data file is incorrect size");
    }

    NumberOfBlobsInFile = FileSizeInBytes/(inputBlobSize + outputBlobSize);

    if(NumberOfBlobsInFile < numberOfBlobsInBuffer)
    {
        throw SOM_EXCEPTION("Data file is too small, there are less examples (" + std::to_string(NumberOfBlobsInFile) + ") then a single buffer " + std::to_string(numberOfBlobsInBuffer));
    }

    Buffer.resize(numberOfBlobsInBuffer*(inputBlobSize+outputBlobSize));
    NextBlobIndex = 0;
    NextBlobInFileIndex = 0;
    File.seekg(0); //Go back to start

    FillBuffer();
}

bool SequentialFileDataLoader::ReadBlob(char* inputBufferAddress, char* outputBufferAddress)
{
    std::memcpy(inputBufferAddress, &Buffer[NextBlobIndex*(InputBlobSize+OutputBlobSize)], InputBlobSize);
    std::memcpy(outputBufferAddress, &Buffer[NextBlobIndex*(InputBlobSize+OutputBlobSize) + InputBlobSize], OutputBlobSize);

    NextBlobIndex++;
    bool passed_end_of_file = (NextBlobInFileIndex + NextBlobIndex) == NumberOfBlobsInFile;

    if(NextBlobIndex >= NumberOfBlobsInBuffer)
    {
        //We've accessed all blobs in the file, so signal epoc end and refill the buffer
        NextBlobIndex = 0;
        FillBuffer();
    }

    return passed_end_of_file;
}

int64_t SequentialFileDataLoader::GetInputDataSize() const
{
    return InputBlobSize;
}

int64_t SequentialFileDataLoader::GetExpectedOutputDataSize() const
{
    return OutputBlobSize;
}

void SequentialFileDataLoader::FillBuffer()
{
    for(int64_t buffer_index = 0; buffer_index < NumberOfBlobsInBuffer; )
    {
        int64_t number_of_blobs_left_in_file = NumberOfBlobsInFile - NextBlobInFileIndex;
        int64_t number_of_blobs_left_to_fill = NumberOfBlobsInBuffer - buffer_index;

        if(number_of_blobs_left_in_file > number_of_blobs_left_to_fill)
        {
            File.read(((char *) &(Buffer[buffer_index*(InputBlobSize+OutputBlobSize)])), (OutputBlobSize+InputBlobSize)*number_of_blobs_left_to_fill);
            NextBlobInFileIndex += number_of_blobs_left_to_fill;
            buffer_index += number_of_blobs_left_to_fill;
        }
        else
        {
            File.read(((char *) &(Buffer[buffer_index*(InputBlobSize+OutputBlobSize)])), (OutputBlobSize+InputBlobSize)*number_of_blobs_left_in_file);
            buffer_index += number_of_blobs_left_in_file;
            File.seekg(0); //Go back to start
            NextBlobInFileIndex = 0;
        }
    }
}
