#pragma once

#include "SQLite3Wrapper.hpp"
#include<memory>
#include<vector>
#include<map>

namespace GoodBot
{

class LogEntry
{
public:
    void clear();

    std::string ExperimentGroupName;
    std::string DataSetName;
    std::string InvestigationMethod;
    double BestTestError;
    double FinalTestError;
    int64_t TimeOfCompletion;
    int64_t RunTime;
    int64_t NumberOfTrainingIterations;
    int64_t NumberOfTrainingEpocs;
    std::string HashOfNetspace;
    std::string NetspaceSummary;

    //Hyperparameter name must be unique
    std::map<std::string, std::vector<int64_t>> IntegerHyperParameters;
    std::map<std::string, std::vector<double>> DoubleHyperParameters;
    std::map<std::string, std::vector<std::string>> StringHyperParameters;
};

class ExperimentLogger
{
  public:
    ExperimentLogger(const std::string& databaseFilePath);

    void AddEntry(const LogEntry& entry);

protected:
    CRWUtility::SQLITE3::DatabaseConnection Session;
    int64_t NextMainEntryKey;

    std::unique_ptr<CRWUtility::SQLITE3::Statement> InsertMainLogEntry;
    std::unique_ptr<CRWUtility::SQLITE3::Statement> InsertIntegerKeyValue;
    std::unique_ptr<CRWUtility::SQLITE3::Statement> InsertDoubleKeyValue;
    std::unique_ptr<CRWUtility::SQLITE3::Statement> InsertStringKeyValue;

    void PrepareInsertionStatements();
};

/**
 * This function attempts to retrieve the entry associated with the given primary key.
 * @param entryPrimaryKey: The primary key of the entry to retrieve
 * @param entryBuffer: A buffer to store the retrieved entry in
 * @param session: The database connection to use
 * @return: True if the entry was retrieved and false otherwise
 */
bool RetrieveEntry(int64_t entryPrimaryKey, LogEntry& entryBuffer, CRWUtility::SQLITE3::DatabaseConnection& session);

/**
 * This function retrieves all entries from the database, ordered by ExperimentId
 * @param session: A connection to the database to retrieve from
 * @return: All of the retrieved log entries
 */
std::vector<LogEntry> RetrieveAllEntries(CRWUtility::SQLITE3::DatabaseConnection& session);


















}
bool operator==(const GoodBot::LogEntry& lhs, const GoodBot::LogEntry& rhs);
