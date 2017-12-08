#include "DataLoader.hpp"
#include<fstream>
#include<vector>

namespace GoodBot
{

class SequentialFileDataLoader : public DataLoader
{
public:
SequentialFileDataLoader(const std::string& sourceFilePath, int64_t inputBlobSize, int64_t outputBlobSize, int64_t numberOfBlobsInBuffer);

virtual bool ReadBlob(char* inputBufferAddress, char* outputBufferAddress) override;

virtual int64_t GetInputDataSize() const override;

virtual int64_t GetExpectedOutputDataSize() const override;

protected:
void FillBuffer();

std::ifstream File;
int64_t FileSizeInBytes;
int64_t InputBlobSize;
int64_t OutputBlobSize;
int64_t NumberOfBlobsInBuffer;
int64_t NumberOfBlobsInFile;

int64_t NextBlobInFileIndex;
int64_t NextBlobIndex;

std::vector<char> Buffer;
};






















}
