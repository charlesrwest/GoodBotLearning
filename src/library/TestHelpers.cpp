#include "TestHelpers.hpp"

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
