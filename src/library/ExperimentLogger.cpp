#include "ExperimentLogger.hpp"
#include "SOMException.hpp"

using namespace GoodBot;

void LogEntry::clear()
{
    ExperimentGroupName = "";
    DataSetName = "";
    InvestigationMethod = "";
    BestTestError = 0.0;
    FinalTestError = 0.0;
    TimeOfCompletion = 0;
    RunTime = 0;
    NumberOfTrainingIterations = 0;
    NumberOfTrainingEpocs = 0;
    HashOfNetspace = "";
    NetspaceSummary = "";

    IntegerHyperParameters.clear();
    DoubleHyperParameters.clear();
    StringHyperParameters.clear();
}

bool operator==(const GoodBot::LogEntry& lhs, const GoodBot::LogEntry& rhs)
{
    lhs.ExperimentGroupName == rhs.ExperimentGroupName;
    lhs.DataSetName == rhs.DataSetName;
    lhs.InvestigationMethod == rhs.InvestigationMethod;
    lhs.BestTestError == rhs.BestTestError;
    lhs.FinalTestError == rhs.FinalTestError;
    lhs.TimeOfCompletion == rhs.TimeOfCompletion;
    lhs.RunTime == rhs.RunTime;
    lhs.NumberOfTrainingIterations == rhs.NumberOfTrainingIterations;
    lhs.NumberOfTrainingEpocs == rhs.NumberOfTrainingEpocs;
    lhs.HashOfNetspace == rhs.HashOfNetspace;
    lhs.NetspaceSummary == rhs.NetspaceSummary;
    lhs.IntegerHyperParameters == rhs.IntegerHyperParameters;
    lhs.DoubleHyperParameters == rhs.DoubleHyperParameters;
    lhs.StringHyperParameters == rhs.StringHyperParameters;
}

void CreateTables(CRWUtility::SQLITE3::DatabaseConnection& session)
{
    std::string table_creation_statement =
            "CREATE TABLE IF NOT EXISTS Experiments (ExperimentId integer primary key, DataSetName blob, ExperimentGroupName blob, InvestigationMethod blob, BestTestError double not null, FinalTestError double not null, TimeOfCompletion integer not null, RunTime integer not null, NumberOfTrainingIterations integer not null, NumberOfTrainingEpocs integer not null, HashOfNetspace blob, Netspace blob); \
             CREATE TABLE IF NOT EXISTS IntegerKeyValues (Id integer primary key AUTOINCREMENT, ParameterName blob not null, ExperimentId integer not null, Value integer not null, foreign key(ExperimentId) references Experiments(ExperimentId) on delete cascade);\
             CREATE TABLE IF NOT EXISTS DoubleKeyValues (Id integer primary key AUTOINCREMENT, ParameterName blob not null, ExperimentId integer not null, Value double not null, foreign key(ExperimentId) references Experiments(ExperimentId) on delete cascade);\
             CREATE TABLE IF NOT EXISTS TextKeyValues (Id integer primary key AUTOINCREMENT, ParameterName blob not null, ExperimentId integer not null, Value blob not null, foreign key(ExperimentId) references Experiments(ExperimentId) on delete cascade);";

    bool tables_made = session.ExecuteCommand(table_creation_statement);
    SOM_ASSERT(tables_made, "Could not make tables");
}

int64_t GetNextMainEntryPrimaryKeyFromTable(CRWUtility::SQLITE3::DatabaseConnection& session)
{
    int64_t primary_key_buffer = 0;
    CRWUtility::SQLITE3::Statement statement("SELECT ExperimentId FROM Experiments ORDER BY ExperimentId DESC LIMIT 1;", session);

    SOM_TRY
    statement.Step();
    SOM_CATCH("Error stepping statement")

    if(statement.ResultAvailable())
    {
      SOM_TRY
        statement.Retrieve(0, primary_key_buffer);
      SOM_CATCH("Error retrieving highest experiment id")
    }

    //Increment to avoid collision
    primary_key_buffer++;

    return primary_key_buffer;
}

ExperimentLogger::ExperimentLogger(const std::string& databaseFilePath) : Session(databaseFilePath)
{
    SOM_TRY
    Session.EnableForeignKeyConstraints();
    SOM_CATCH("Error enabling foreign key constraints")

    SOM_TRY
    CreateTables(Session);
    SOM_CATCH("Error making tables")

    SOM_TRY
    NextMainEntryKey = GetNextMainEntryPrimaryKeyFromTable(Session);
    SOM_CATCH("Error getting highest existing primary key from main table")

    SOM_TRY
    PrepareInsertionStatements();
    SOM_CATCH("Error compiling insertion statements")
}

void ExperimentLogger::AddEntry(const LogEntry& entry)
{
    //Copy entry into buffer
    CRWUtility::SQLITE3::TransactionGuard guard(Session);

    //Insert main row
    InsertMainLogEntry->Bind(1, NextMainEntryKey);
    InsertMainLogEntry->BindBlob(2, entry.ExperimentGroupName);
    InsertMainLogEntry->BindBlob(3, entry.DataSetName);
    InsertMainLogEntry->BindBlob(4, entry.InvestigationMethod);
    InsertMainLogEntry->Bind(5, entry.BestTestError);
    InsertMainLogEntry->Bind(6, entry.FinalTestError);
    InsertMainLogEntry->Bind(7, entry.TimeOfCompletion);
    InsertMainLogEntry->Bind(8, entry.RunTime);
    InsertMainLogEntry->Bind(9, entry.NumberOfTrainingIterations);
    InsertMainLogEntry->Bind(10, entry.NumberOfTrainingEpocs);
    InsertMainLogEntry->BindBlob(11, entry.HashOfNetspace);
    InsertMainLogEntry->BindBlob(12, entry.NetspaceSummary);

    SOM_TRY
        InsertMainLogEntry->StepAndReset();
    SOM_CATCH("Error inserting main entry")

    //Insert Hyperparameter values
    InsertIntegerKeyValue->Bind(2, NextMainEntryKey);
    for(const std::pair<std::string, std::vector<int64_t>> integer_key_value : entry.IntegerHyperParameters)
    {
        for(int64_t value : integer_key_value.second)
        {
            InsertIntegerKeyValue->Bind(1, integer_key_value.first);
            InsertIntegerKeyValue->Bind(3, value);
            InsertIntegerKeyValue->StepAndReset();
        }
    }

    InsertDoubleKeyValue->Bind(2, NextMainEntryKey);
    for(const std::pair<std::string, std::vector<double>> double_key_value : entry.DoubleHyperParameters)
    {
        for(double value : double_key_value.second)
        {
            InsertDoubleKeyValue->Bind(1, double_key_value.first);
            InsertDoubleKeyValue->Bind(3, value);
            InsertDoubleKeyValue->StepAndReset();
        }
    }

    InsertStringKeyValue->Bind(2, NextMainEntryKey);
    for(const std::pair<std::string, std::vector<std::string>> string_key_value : entry.StringHyperParameters)
    {
        for(const std::string& value : string_key_value.second)
        {
            InsertStringKeyValue->Bind(1, string_key_value.first);
            InsertStringKeyValue->Bind(3, value);
            InsertStringKeyValue->StepAndReset();
        }
    }

    //Increment the primary key to use for the next insertion
    NextMainEntryKey++;
}

void ExperimentLogger::PrepareInsertionStatements()
{
    SOM_TRY
    InsertMainLogEntry.reset(new CRWUtility::SQLITE3::Statement("INSERT INTO Experiments (ExperimentId, DataSetName, ExperimentGroupName,\
                                                              InvestigationMethod, BestTestError, FinalTestError,\
                                                             TimeOfCompletion, RunTime, NumberOfTrainingIterations,\
                                                             NumberOfTrainingEpocs, HashOfNetspace, Netspace)\
                                                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", Session));
    SOM_CATCH("Error making insert statement for main table\n")

    SOM_TRY
    InsertIntegerKeyValue.reset(new CRWUtility::SQLITE3::Statement("INSERT INTO IntegerKeyValues (ParameterName, ExperimentId, Value) VALUES(?, ?, ?);", Session));
    SOM_CATCH("Error making insert statement for integer key values\n")

    SOM_TRY
    InsertDoubleKeyValue.reset(new CRWUtility::SQLITE3::Statement("INSERT INTO DoubleKeyValues (ParameterName, ExperimentId, Value) \
                                        VALUES(?, ?, ?);", Session));
    SOM_CATCH("Error making insert statement for double key values\n")

    SOM_TRY
    InsertStringKeyValue.reset(new CRWUtility::SQLITE3::Statement("INSERT INTO TextKeyValues (ParameterName, ExperimentId, Value) \
                                        VALUES(?, ?, ?);", Session));
    SOM_CATCH("Error making insert statement for string key values\n")
}

bool GoodBot::RetrieveEntry(int64_t entryPrimaryKey, LogEntry& entryBuffer, CRWUtility::SQLITE3::DatabaseConnection& session)
{
    SOM_TRY
    CRWUtility::SQLITE3::Statement retrieve_primary_row("SELECT DataSetName, ExperimentGroupName, InvestigationMethod, BestTestError, FinalTestError,\
 TimeOfCompletion, RunTime, NumberOfTrainingIterations, NumberOfTrainingEpocs, HashOfNetspace, Netspace FROM Experiments WHERE ExperimentId=?;", session);

    retrieve_primary_row.Bind(1, entryPrimaryKey);

    retrieve_primary_row.Step();

    if(!retrieve_primary_row.ResultAvailable())
    {
        return false;
    }

    entryBuffer.clear();
    retrieve_primary_row.RetrieveBlob(0, entryBuffer.DataSetName);
    retrieve_primary_row.RetrieveBlob(1, entryBuffer.ExperimentGroupName);
    retrieve_primary_row.RetrieveBlob(2, entryBuffer.InvestigationMethod);
    retrieve_primary_row.Retrieve(3, entryBuffer.BestTestError);
    retrieve_primary_row.Retrieve(4, entryBuffer.FinalTestError);
    retrieve_primary_row.Retrieve(5, entryBuffer.TimeOfCompletion);
    retrieve_primary_row.Retrieve(6, entryBuffer.RunTime);
    retrieve_primary_row.Retrieve(7, entryBuffer.NumberOfTrainingIterations);
    retrieve_primary_row.Retrieve(8, entryBuffer.NumberOfTrainingEpocs);
    retrieve_primary_row.RetrieveBlob(9, entryBuffer.HashOfNetspace);
    retrieve_primary_row.RetrieveBlob(10, entryBuffer.NetspaceSummary);
    SOM_CATCH("Error getting primary row\n")


    SOM_TRY
    CRWUtility::SQLITE3::Statement retrieve_integer_hyper_parameters("SELECT ParameterName, Value FROM IntegerKeyValues WHERE ExperimentId=?;", session);

    retrieve_integer_hyper_parameters.Bind(1, entryPrimaryKey);

    retrieve_integer_hyper_parameters.Step();
    while(retrieve_integer_hyper_parameters.ResultAvailable())
    {
        std::string key;
        int64_t value = 0;

        retrieve_integer_hyper_parameters.RetrieveBlob(0, key);
        retrieve_integer_hyper_parameters.Retrieve(1, value);

        entryBuffer.IntegerHyperParameters[key].emplace_back(value);

        retrieve_integer_hyper_parameters.Step();
    }
    SOM_CATCH("Error getting integer hyper parameters\n")

    SOM_TRY
    CRWUtility::SQLITE3::Statement retrieve_double_hyper_parameters("SELECT ParameterName, Value FROM DoubleKeyValues WHERE ExperimentId=?;", session);

    retrieve_double_hyper_parameters.Bind(1, entryPrimaryKey);

    retrieve_double_hyper_parameters.Step();
    while(retrieve_double_hyper_parameters.ResultAvailable())
    {
        std::string key;
        double value = 0;

        retrieve_double_hyper_parameters.RetrieveBlob(0, key);
        retrieve_double_hyper_parameters.Retrieve(1, value);

        entryBuffer.DoubleHyperParameters[key].emplace_back(value);

        retrieve_double_hyper_parameters.Step();
    }
    SOM_CATCH("Error getting double hyper parameters\n")

    SOM_TRY
    CRWUtility::SQLITE3::Statement retrieve_string_hyper_parameters("SELECT ParameterName, Value FROM TextKeyValues WHERE ExperimentId=?;", session);

    retrieve_string_hyper_parameters.Bind(1, entryPrimaryKey);

    retrieve_string_hyper_parameters.Step();
    while(retrieve_string_hyper_parameters.ResultAvailable())
    {
        std::string key;
        std::string value;

        retrieve_string_hyper_parameters.RetrieveBlob(0, key);
        retrieve_string_hyper_parameters.RetrieveBlob(1, value);

        entryBuffer.StringHyperParameters[key].emplace_back(value);

        retrieve_string_hyper_parameters.Step();
    }
    SOM_CATCH("Error getting string hyper parameters\n")

    return true;
}

std::vector<LogEntry> GoodBot::RetrieveAllEntries(CRWUtility::SQLITE3::DatabaseConnection& session)
{
    std::vector<int64_t> primary_keys;
    SOM_TRY
    CRWUtility::SQLITE3::Statement entry_primary_keys("SELECT ExperimentId FROM Experiments;", session);

    int64_t key_buffer;
    entry_primary_keys.Step();
    while(entry_primary_keys.ResultAvailable())
    {
        entry_primary_keys.Retrieve(0, key_buffer);
        primary_keys.emplace_back(key_buffer);
        entry_primary_keys.Step();
    }
    SOM_CATCH("Error retrieving primary key list\n")

    std::vector<LogEntry> retrieved_entries;
    LogEntry entry_buffer;
    for(int64_t primary_key : primary_keys)
    {
        bool retrieved = RetrieveEntry(primary_key, entry_buffer, session);

        if(retrieved)
        {
            retrieved_entries.emplace_back(entry_buffer);
        }
    }

    return retrieved_entries;
}
