#pragma once

#include<random>
#include<fstream>
#include<queue>
#include "DataLoader.hpp"

namespace GoodBot
{

//Size of input blobs, size of output blobs, number of blobs in buffer, number of buffers
class RandomizedFileDataLoader : public DataLoader
{
public:
RandomizedFileDataLoader(const std::string& sourceFilePath, int64_t inputBlobSize, int64_t outputBlobSize,
                         int64_t numberOfBlobsPerBuffer, int64_t numberOfBuffers);

virtual bool ReadBlob(char* inputBufferAddress, char* outputBufferAddress) override;

virtual int64_t GetInputDataSize() const override;

virtual int64_t GetExpectedOutputDataSize() const override;

protected:
//Seek to a random place in the file and fill a buffer
bool FillBuffersIfEmpty(); //Returns true if all blobs in file were explored before
void FillBuffer(int64_t bufferIndex);
void FillFileOffsets();
int64_t GetRandomNonEmptyBufferIndex();


std::random_device RandomDevice;
std::mt19937 Randomness;

std::ifstream File;
int64_t FileSizeInBytes;
int64_t InputBlobSize;
int64_t OutputBlobSize;
int64_t NumberOfBlobsPerBuffer;

std::queue<int64_t> FileOffsets; //Remaining places in the file to read from
std::vector<std::queue<int64_t>> RandomBlobIndexBuffers;

std::vector<std::vector<char>> Buffers;
};
















}
