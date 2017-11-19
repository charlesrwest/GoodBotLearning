#include "TestHelpers.hpp"

void print(const google::protobuf::Message& inputMessage)
{
std::string buffer;

google::protobuf::TextFormat::PrintToString(inputMessage, &buffer);

std::cout << buffer<<std::endl;
};

bool BlobNamesFound(const std::vector<std::string>& blobNames, const caffe2::Workspace& workspace)
{
std::vector<std::string> current_blobs = workspace.Blobs();

for(const std::string& blob_name : blobNames)
{
if(std::find(current_blobs.begin(), current_blobs.end(), blob_name) == current_blobs.end())
{
return false;
}
}

return true;
}

bool BlobShapeMatches(const std::string& blobName, const std::vector<int64_t>& expectedShape, const caffe2::Workspace& workspace)
{
caffe2::TensorCPU tensor = GoodBot::GetTensor(*workspace.GetBlob(blobName));

return expectedShape == tensor.dims();
}

double SimpleTestObjectiveFunction(const std::vector<double>& input, std::vector<double>& gradient, void* userData)
{
if(!gradient.empty())
{
gradient[0] = 0.0;
gradient[1] = 0.5 / sqrt(input[1]);
}

return sqrt(input[1]);
}

double SimpleTestVConstraint(const std::vector<double>& input, std::vector<double>& gradient, void* userData)
{
const SimpleTestConstraintData* constraint_data = reinterpret_cast<const SimpleTestConstraintData*>(userData);
double a = constraint_data->a;
double b = constraint_data->b;

if(!gradient.empty())
{
gradient[0] = 3.0*a*(a*input[0]+b)*(a*input[0]+b);
gradient[1] = -1.0;
}

return ((a*input[0]+b)*(a*input[0]+b)*(a*input[0]+b) - input[1]);
}

void AddDataToVector(const PseudoImage<char>& data, std::vector<char>& outputBuffer)
{
    const char* data_pointer = data.GetData();
    int64_t data_size = data.GetSize();

    for(int64_t data_index = 0; data_index < data_size; data_index++)
    {
        outputBuffer.emplace_back((data_pointer)[data_index]);
    }
}

std::pair<std::vector<char>, std::vector<char>> SplitDataSet(double fractionInFirstSet, int64_t exampleSizeInBytes, const std::vector<char>& dataSet)
{
    SOM_ASSERT((dataSet.size() % exampleSizeInBytes) == 0, "Dataset cannot be cleanly split into examples");

    int64_t number_of_examples_in_dataset = dataSet.size()/exampleSizeInBytes;

    int64_t number_of_examples_in_first_set = number_of_examples_in_dataset*fractionInFirstSet;
    int64_t number_of_examples_in_second_set = number_of_examples_in_dataset - number_of_examples_in_first_set;

    std::pair<std::vector<char>, std::vector<char>> result;

    //Resize vector to required size and then copy the associated data over
    result.first.resize(number_of_examples_in_first_set*exampleSizeInBytes);
    memcpy(&result.first[0], &dataSet[0], result.first.size());

    result.second.resize(number_of_examples_in_second_set*exampleSizeInBytes);
    memcpy(&result.second[0], &dataSet[number_of_examples_in_first_set*exampleSizeInBytes], result.second.size());

    return result;
}
