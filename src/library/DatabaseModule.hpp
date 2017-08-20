#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{


struct DatabaseModuleParameters
{
std::string ModuleName;
std::string TrainingDatabaseSourceName;
std::string TrainingDatabaseType;
int64_t BatchSize = 1; //0 means load all
};

//A pre-initialized DB reader. Typically, this is obtained by calling CreateDB operator with a db_name and a db_type. The resulting output blob is a DB Reader tensor

/**
This class is an implementation of a database data sources.  Which database it is pulling from can be switched based on mode.
*/
class DatabaseModule : public ComputeModuleDefinition
{
public:
DatabaseModule(const DatabaseModuleParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

std::string GetDatabaseOutputBlobName() const;

private:
std::string TrainingDatabaseSourceName;
std::string TrainingDatabaseType;
int64_t BatchSize; //0 means load all
};









}
