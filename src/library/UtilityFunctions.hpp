#pragma once

#include<string>
#include<vector>
#include "caffe2/proto/caffe2.pb.h"

namespace GoodBot
{

/**
  This function takes a file which consists of a sequence of (input/output) pairs which together for examples and splits it into 2 files.
  @param fractionInFirst: What fraction of the examples should end up in the first file.
  @param exampleSize: The size of each example (input/ouput pair) in bytes
  @param batchSize: The number of examples in each batch (if needed, examples will be discarded so both files are a multiple of the batch size)
  @param originalFilePath: The path to the file to split
  @param firstFilePath: The path of the first output file
  @param secondFilePath: The path of the second output file
  */
void SplitBlobberFile(double fractionInFirst, int64_t exampleSize, int64_t batchSize,
                      const std::string& originalFilePath, const std::string& firstFilePath, const std::string& secondFilePath);

/**
This function returns true if the the pattern is found at the beginning of the string.
@param pattern: The pattern to check for
@param text: The text to check for the pattern in
*/
bool StringAtStart(const std::string& pattern, const std::string& text);

/**
This function generates a (capitalized) hex string using rand() to select digits.
@inputLength:How many digits the string should have
@return: The generated string
*/
std::string GenerateRandomHexString(int64_t inputLength);

/**
This function returns a string indicating what a gradient blob automatically generated from the given blob name would be named.
@param inputOperatorBlobName: The name to create a gradient name for
@return: The generated name
*/
std::string MakeGradientOperatorBlobName(const std::string& inputOperatorBlobName);

/**
This function attempts to make a gradient computation operator for the given network operator.  Generated operators can fail when run if the given operator does not support having gradients made for it.
@param inputOperator: The operator to make the gradient for
@return: The generated gradient creating operators.
*/
std::vector<caffe2::OperatorDef> GetGradientOperatorsFromOperator(const caffe2::OperatorDef& inputOperator);

/**
 * Some have gradients that just pass through (such as "Sum"), so sum_input_grad_names should be set to sum_output_grad_name.
 * @param inputOperator: The operator to check for replacements
 * @return: The list of blobs to replace (first is replaced by second)
 */
std::vector<std::pair<std::string, std::string>> GetGradientNamesToReplace(const caffe2::OperatorDef& inputOperator);

/**
Network operators can create blobs by having them as outputs.  However, caffe2 throws an error if it encounters an operator which references an input that hasn't been made yet as it works through a network definition (even if it is defined as an output in one of the later operators).  This function was created to prevent these sorts of failures.  It analyzes which blobs create which outputs and reorders the network operators (if possible) to prevent operators from being in the list before the operators which create what they read.  If there are no problems with the given operator orders or it cannot find a workable order, the original list is returned.
@param inputOperators: The operators to reorder.
@param inputExistingBlobNames: A list of blobs which already exist for some other reason
@return: The reordered operator list.
*/
std::vector<caffe2::OperatorDef> ReorderOperatorsToResolveDependencies(const std::vector<caffe2::OperatorDef>& inputOperators, const std::vector<std::string>& inputExistingBlobNames);

std::string MakeWeightBlobName(const std::string& prefix);

std::string MakeBiasBlobName(const std::string& prefix);

 









}
