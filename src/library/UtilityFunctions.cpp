#include "UtilityFunctions.hpp"

#include<cstdlib>
#include "caffe2/core/operator_gradient.h"
#include<fstream>
#include "SOMException.hpp"

using namespace GoodBot;

void GoodBot::SplitBlobberFile(double fractionInFirst, int64_t exampleSize, int64_t batchSize,
                      const std::string& originalFilePath, const std::string& firstFilePath, const std::string& secondFilePath)
{
    //Open at end of file
    std::ifstream original_file_stream(originalFilePath, std::ifstream::binary | std::ifstream::ate);

    int64_t original_file_size = original_file_stream.tellg();
    SOM_ASSERT((original_file_size % exampleSize) == 0, "File size does not evenly divide by example size");
    int64_t number_of_examples_in_original_file = original_file_size / exampleSize;
    int64_t number_of_batches_in_original_file = number_of_examples_in_original_file / batchSize; //Truncate off examples which are not multiples of a batch size

    int64_t number_of_batches_for_first_file = number_of_batches_in_original_file*fractionInFirst;
    int64_t number_of_batches_for_second_file = number_of_batches_in_original_file - number_of_batches_for_first_file;

    //Set back to start
    original_file_stream.seekg(0);

    std::vector<char> read_buffer(exampleSize*batchSize);

    std::ofstream first_file_stream(firstFilePath, std::ofstream::binary);
    std::ofstream second_file_stream(secondFilePath, std::ofstream::binary);
    std::uniform_real_distribution<double> file_chooser(0.0, 1.0);
    std::random_device randomness;
    int64_t number_of_examples_in_first_file = 0;
    int64_t number_of_examples_in_second_file = 0;

    for(int64_t example_index = 0; example_index < (batchSize*(number_of_batches_for_first_file+number_of_batches_for_second_file)); example_index++)
    {
        bool first_file_full = number_of_examples_in_first_file >= (number_of_batches_for_first_file*batchSize);
        bool second_file_full = number_of_examples_in_second_file >= (number_of_batches_for_second_file*batchSize);

        //Distribute between the two files randomly
        original_file_stream.read(&read_buffer[0], exampleSize);
        if((second_file_full || (file_chooser(randomness) <= fractionInFirst)) && (!first_file_full))
        {
            first_file_stream.write(&read_buffer[0], exampleSize);
            number_of_examples_in_first_file++;
        }
        else if(!second_file_full)
        {
            second_file_stream.write(&read_buffer[0], exampleSize);
            number_of_examples_in_second_file++;
        }

        //If both files are full, just skip
    }

}

bool GoodBot::StringAtStart(const std::string& pattern, const std::string& text)
{
return text.find(pattern) == 0;
}

/**
This function generates a (capitalized) hex string using rand() to select digits.
@inputLength:How many digits the string should have
@return: The generated string
*/
std::string GoodBot::GenerateRandomHexString(int64_t inputLength)
{
const static std::string lookUpTable("0123456789ABCDEF");

std::string result;

while(result.size() < inputLength)
{
result.push_back(lookUpTable[rand() % lookUpTable.size()]);
}

return result;
}

std::string GoodBot::MakeGradientOperatorBlobName(const std::string& inputOperatorBlobName)
{
return inputOperatorBlobName + "_grad";
}

std::vector<caffe2::OperatorDef> GoodBot::GetGradientOperatorsFromOperator(const caffe2::OperatorDef& inputOperator)
{
std::vector<caffe2::GradientWrapper> gradientBlobNames;
for(int64_t outputIndex = 0; outputIndex < inputOperator.output_size(); outputIndex++)
{
caffe2::GradientWrapper wrapper;
wrapper.dense_ = MakeGradientOperatorBlobName(inputOperator.output(outputIndex));

gradientBlobNames.emplace_back(wrapper);
}

caffe2::GradientOpsMeta operatorsAndWrappers = caffe2::GetGradientForOp(inputOperator, gradientBlobNames);

for(caffe2::OperatorDef& gradientOperator : operatorsAndWrappers.ops_)
{
gradientOperator.set_is_gradient_op(true); //Mark as gradient operator
}

return operatorsAndWrappers.ops_;
}

std::vector<caffe2::OperatorDef> GoodBot::ReorderOperatorsToResolveDependencies(const std::vector<caffe2::OperatorDef>& inputOperators, const std::vector<std::string>& inputExistingBlobNames)
{
//Inefficient, but doesn't really matter all that much at this scale
std::vector<caffe2::OperatorDef> results;

using v_size_t = std::vector<caffe2::OperatorDef>::size_type;

std::set<std::string> availableInputBlobNames;
availableInputBlobNames.insert(inputExistingBlobNames.begin(), inputExistingBlobNames.end());

std::set<v_size_t> processedEntryIndices;

v_size_t numberOfEntriesRemovedThisLoop = 1;  //Initialize to non-zero to prevent exit at start of loop

std::function<bool(const caffe2::OperatorDef&)> RequiredInputBlobsAreAvailable = [&](const caffe2::OperatorDef& inputOperator)
{
for(int64_t inputIndex = 0; inputIndex < inputOperator.input_size(); inputIndex++)
{
if(availableInputBlobNames.count(inputOperator.input(inputIndex)) == 0)
{
return false; //This operator requires an input that is not available yet
}
}

return true; //All input requirements are satisfied
};

std::function<void(v_size_t)> AddOperatorToResults = [&](v_size_t inputOperatorIndex)
{
const caffe2::OperatorDef& operatorToAdd = inputOperators[inputOperatorIndex];

results.emplace_back(operatorToAdd);
//Add associated outputs to the available blob pool
for(int64_t outputIndex = 0; outputIndex < operatorToAdd.output_size(); outputIndex++)
{
availableInputBlobNames.emplace(operatorToAdd.output(outputIndex));
}

processedEntryIndices.emplace(inputOperatorIndex);
numberOfEntriesRemovedThisLoop++;
};

//Loop until we run out of things to process, adding operators that have their input dependencies resolved and adding their outputs to the pool of available inputs
while((numberOfEntriesRemovedThisLoop > 0) && (results.size() < inputOperators.size()))
{
numberOfEntriesRemovedThisLoop = 0;
for(size_t operatorIndex = 0; operatorIndex < inputOperators.size(); operatorIndex++)
{
const caffe2::OperatorDef& currentOperator = inputOperators[operatorIndex];

if(processedEntryIndices.count(operatorIndex) > 0)
{
continue; //We already have this entry, so skip
}

if(RequiredInputBlobsAreAvailable(currentOperator))
{
AddOperatorToResults(operatorIndex);
}
}

}

if(processedEntryIndices.size() != inputOperators.size())
{
    std::cout << "Could not resolve dependencies for some operators: " << std::endl;
for(int64_t operator_index = 0; operator_index < inputOperators.size(); operator_index++)
{
    if(processedEntryIndices.count(operator_index) > 0)
    {
    continue; //We already have this entry, so skip
    }

    const caffe2::OperatorDef& currentOperator = inputOperators[operator_index];
    for(int64_t inputIndex = 0; inputIndex < currentOperator.input_size(); inputIndex++)
    {
    if(availableInputBlobNames.count(currentOperator.input(inputIndex)) == 0)
    {
    std::cout << currentOperator.name() << " is missing required input " << currentOperator.input(inputIndex) << " but would produce "; //This operator requires an input that is not available yet
    for(int64_t output_index = 0; output_index < currentOperator.output_size(); output_index++)
    {
        std::cout << currentOperator.output(output_index) << " ";
    }
    std::cout << std::endl;

    }
    }
}
    std::cout << "Available inputs: " << std::endl;
    for(const std::string& available_input_name : availableInputBlobNames)
    {
        std::cout << available_input_name << std::endl;
    }
}

SOM_ASSERT(processedEntryIndices.size() == inputOperators.size(), "Some operators were not able to be resolved");

return results;
}

std::string GoodBot::MakeWeightBlobName(const std::string& prefix)
{
return prefix + "_w";
}

std::string GoodBot::MakeBiasBlobName(const std::string& prefix)
{
return prefix + "_b";
}
