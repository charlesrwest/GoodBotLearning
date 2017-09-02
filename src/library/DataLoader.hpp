#pragma once

#include<random>
#include<fstream>
#include<queue>

namespace GoodBot
{

//Size of input blobs, size of output blobs, number of blobs in buffer, number of buffers
class DataLoader
{
public:
DataLoader(const std::string& sourceFilePath, int64_t inputBlobSize, int64_t outputBlobSize, int64_t numberOfBlobsPerBuffer, int64_t numberOfBuffers, int64_t maxRereadsBeforeRefill);

void ReadBlobs(char* inputBufferAddress, char* outputBufferAddress, int64_t numberOfBlobs); 
void ReadBlob(char* inputBufferAddress, char* outputBufferAddress); 


protected:
//Seek to a random place in the file and fill a buffer
void FillBuffer(int64_t bufferIndex);

std::random_device RandomDevice;
std::mt19937 Randomness;
std::uniform_int_distribution<int64_t> BufferSelector;
std::uniform_int_distribution<int64_t> FileBlobIndexSelector;

std::ifstream File;
int64_t FileSizeInBytes;
int64_t InputBlobSize;
int64_t OutputBlobSize;
int64_t NumberOfBlobsPerBuffer;
int64_t MaxRereadsBeforeRefill;  //How many times to read each example in a buffer before refilling the buffer

std::vector<std::queue<int64_t>> RandomBlobIndexBuffers;

std::vector<std::vector<char>> InputBuffers;
std::vector<std::vector<char>> OutputBuffers;
};
















}
