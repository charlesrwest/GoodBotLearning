#include "DatabaseModule.hpp"


using namespace GoodBot;




DatabaseModule::DatabaseModule(const DatabaseModuleParameters& inputParameters) : TrainingDatabaseSourceName(inputParameters.TrainingDatabaseSourceName), TrainingDatabaseType(inputParameters.TrainingDatabaseType), BatchSize(inputParameters.BatchSize)
{
SetName(inputParameters.ModuleName);
}


std::vector<std::string> DatabaseModule::GetInputBlobNames() const
{
return {};
}

std::vector<std::string> DatabaseModule::GetOutputBlobNames() const
{
return {GetDatabaseOutputBlobName()};
}

std::vector<caffe2::OperatorDef> DatabaseModule::GetNetworkOperators() const
{
if(Mode() == "TRAIN")
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& database = results.back();

database.set_type("TensorProtosDBInput");

caffe2::Argument& db_type = *database.add_arg();
db_type.set_name("batch_size");
db_type.add_ints(BatchSize);

database.add_input(TrainingDatabaseSourceName);
database.add_output(GetDatabaseOutputBlobName());

return results;
}

}

std::vector<caffe2::OperatorDef> DatabaseModule::GetNetworkInitializationOperators() const
{
if(Mode() == "TRAIN")
{
std::vector<caffe2::OperatorDef> results;
results.emplace_back();
caffe2::OperatorDef& database_creator = results.back();

database_creator.set_type("CreateDB");

caffe2::Argument& db_type = *database_creator.add_arg();
db_type.set_name("db_type");
db_type.add_strings(TrainingDatabaseType);

caffe2::Argument& db_source = *database_creator.add_arg();
db_type.set_name("db");
db_type.add_strings(TrainingDatabaseSourceName);

return results;
}

return {};
}

std::string DatabaseModule::GetDatabaseOutputBlobName() const
{
return Name() + "_output";
}
