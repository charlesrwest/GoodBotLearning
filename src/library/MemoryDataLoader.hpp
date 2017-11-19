#include "DataLoader.hpp"
#include "TestHelpers.hpp"

namespace GoodBot
{

class MemoryDataLoader : public DataLoader
{
  public:    
    MemoryDataLoader(int64_t numberOfExamples, const char* inputData, int64_t inputExampleSize,
                     const char* expectedOutputData, int64_t expectedOutputExampleSize);

    MemoryDataLoader(int64_t numberOfExamples, const std::vector<char>& inputData, const std::vector<char>& expectedOutputData);

    virtual bool ReadBlob(char* inputBufferAddress, char* expectedOutputBufferAddress) override;

    virtual int64_t GetInputDataSize() const override;

    virtual int64_t GetExpectedOutputDataSize() const override;

    int64_t GetNumberOfExamples() const;

protected:
    std::mt19937_64 mersenne;

    int64_t size_of_input;
    int64_t size_of_output;

    int64_t blob_index_index;
    std::vector<int64_t> blob_indices;

    std::vector<char> input_data;
    std::vector<char> expected_output_data;

    void Initialize(int64_t numberOfExamples, const char* inputData, int64_t inputExampleSize,
                     const char* expectedOutputData, int64_t expectedOutputExampleSize);
    void ResetExampleIndexIndexAndShuffleIndices();
};
























}
