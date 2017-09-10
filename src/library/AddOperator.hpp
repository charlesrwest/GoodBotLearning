#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

class AddOperator : public ComputeModuleDefinition
{
public:
AddOperator(const std::string& name, const std::string& inputBlobName, float scalarValue);

AddOperator(const std::string& name, const std::string& lhsBlobName, const std::string& rhsBlobName, bool broadcasting);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

protected:
std::string LHSBlobName;
std::string RHSBlobName;
bool Broadcasting;

bool GenerateBlob;
float ScalarValue;
};

}
