#pragma once
#include<map>
#include "sqlite3.h"
#include<cassert>
#include "SOMException.hpp"

namespace CRWUtility
{

namespace SQLITE3
{
class DatabaseConnection
{
public:
/**
This function creates an independent in-memory SQLite3 database and a connection to it.
@throws: This function can throw exceptions
*/
DatabaseConnection(); //Makes in memory database

/**
This function connects to or creates a file based SQLite database.
@param inputConnectionString:  The SQLite3 file to connect to 
@throws: This function can throw exceptions
*/
DatabaseConnection(const std::string& inputConnectionString);

/**
This function turns on foreign key constraints for this connection (off by default).
@throws: This function can throw exceptions
*/
void EnableForeignKeyConstraints();

/**
Returns true if the database existed prior to making this connection.
@return: True if the database previously existed (false otherwise)
*/
bool DatabaseExistedPreviously() const;

/**
Return the underlying SQLite3 connection for use.
@return: The SQLite3 connection
*/
sqlite3& GetConnection();

/**
This function lets you execute an arbitrary SQL command which has no parameters.
@param inputSQLString: The command to execute
@return: true if successful and false otherwise
*/
bool ExecuteCommand(const std::string& inputSQLString);

/**
This function releases the database connection.
*/
~DatabaseConnection();

bool databaseExisted;
sqlite3 *connection;
};

class Statement
{
public:
/**
This function compiles a SQLITE3 statement from the given SQL string with the given database connection.
@param inputSQLStatement: The statement to compile
@param inputDatabaseConnection: The database connection to use
@throws: This function can throw exceptions
*/
Statement(const std::string& inputSQLStatement, DatabaseConnection& inputDatabaseConnection);

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Bind(int64_t inputPosition, int64_t inputValue);

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Bind(int64_t inputPosition, double inputValue);

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Bind(int64_t inputPosition, bool inputValue);

/**
This statement binds a value to a particular position in the compiled statement.  This particular instance binds text.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Bind(int64_t inputPosition, const std::string& inputValue);

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void BindBlob(int64_t inputPosition, const std::string& inputValue);

/**
This statement binds null to a particular position in the compiled statement.
@param inputPosition: The position to set to null
@throws: This function throws an exception if the operation fails
*/
void Bind(int64_t inputPosition);

/**
This function returns true if the stepped query has more rows left to return.
@return: True if there is more data to retrieve
*/
bool ResultAvailable() const;

/**
This function retrieves the value stored from an executed query at the given position.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Retrieve(int64_t inputPosition, int64_t& inputResultBuffer);

/**
This function retrieves the value stored from an executed query at the given position.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Retrieve(int64_t inputPosition, double& inputResultBuffer);

/**
This function retrieves the value stored from an executed query at the given position.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Retrieve(int64_t inputPosition, bool& inputResultBuffer);

/**
This function retrieves the value stored from an executed query at the given position.  This particular instance retrieve text.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Retrieve(int64_t inputPosition, std::string& inputResultBuffer);

/**
This function retrieves the value stored from an executed query at the given position.  This particular instance retrieve a blob.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool RetrieveBlob(int64_t inputPosition, std::string& inputResultBuffer);

/**
This function steps the statement to get the data or perform the operation.
@throws: This function throws an exception if the operation fails
*/
void Step();

/**
This function resets the statement so that the operation can be done with new parameters.
@throws: This function throws an exception if the operation fails
*/
void Reset();

/**
This function steps the statement and then resets it
@throws: This function throws an exception if the operations fail
*/
void StepAndReset();

/**
This function calls finalize on the associated compiled statement.
*/
~Statement();

sqlite3_stmt *compiledStatement;
bool ResultIsAvailable;
}; 

class TransactionGuard
{
public:
/**
This function calls "BEGIN TRANSACTION;" on the connection when it is created and "END TRANSACTION;" when it is destroyed.
@param inputDatabaseConnection: The connection to the database to use.
*/
TransactionGuard(DatabaseConnection& inputDatabaseConnection);

/**
Calls "END TRANSACTION;"
*/
~TransactionGuard();

DatabaseConnection* connection;
};


/**
This function creates an independent in-memory SQLite3 database and a connection to it.
@throws: This function can throw exceptions
*/
DatabaseConnection::DatabaseConnection()
{
databaseExisted = false;
if(sqlite3_open_v2(":memory:", &connection, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE,NULL) != SQLITE_OK)
{
throw SOMException("Error creating in memory database\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This function connects to or creates a file based SQLite database.
@param inputConnectionString:  The SQLite3 file to connect to
@throws: This function can throw exceptions 
*/
DatabaseConnection::DatabaseConnection(const std::string& inputConnectionString)
{
if(sqlite3_open_v2(inputConnectionString.c_str(), &connection, SQLITE_OPEN_READWRITE,NULL) == SQLITE_OK)
{
databaseExisted = true;
}
//We couldn't connect to it without making it, so now we will try making it
else if(sqlite3_open_v2(inputConnectionString.c_str(), &connection, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE,NULL) != SQLITE_OK)
{
throw SOMException("Error creating in file based database\n", SQLITE3_ERROR, __FILE__, __LINE__);
databaseExisted = false;
}
}

/**
This function turns on foreign key constraints for this connection (off by default).
@throws: This function can throw exceptions
*/
void DatabaseConnection::EnableForeignKeyConstraints()
{
Statement pragmaStatement("PRAGMA foreign_keys = on;", *this);
pragmaStatement.Step();
}

/**
Returns true if the database existed prior to making this connection.
@return: True if the database previously existed (false otherwise)
*/
bool DatabaseConnection::DatabaseExistedPreviously() const
{
return databaseExisted;
}

/**
Return the underlying SQLite3 connection for use.
@return: The SQLite3 connection
*/
sqlite3& DatabaseConnection::GetConnection()
{
return *connection;
}

/**
This function lets you execute an arbitrary SQL command which has no parameters.
@param inputSQLString: The command to execute
@return: true if successful and false otherwise
*/
bool DatabaseConnection::ExecuteCommand(const std::string& inputSQLString)
{
return sqlite3_exec(connection, inputSQLString.c_str(), NULL, NULL, NULL) == SQLITE_OK;
}

/**
This function releases the database connection.
*/
DatabaseConnection::~DatabaseConnection()
{
bool closeCompleted = sqlite3_close_v2(connection) == SQLITE_OK;
assert(closeCompleted);
}

/**
This function compiles a SQLITE3 statement from the given SQL string with the given database connection.
@param inputSQLStatement: The statement to compile
@param inputDatabaseConnection: The database connection to use
@throws: This function can throw exceptions
*/
Statement::Statement(const std::string& inputSQLStatement, DatabaseConnection& inputDatabaseConnection) : ResultIsAvailable(false)
{
    int return_value = sqlite3_prepare_v2(&inputDatabaseConnection.GetConnection(), inputSQLStatement.c_str(), inputSQLStatement.size(), &compiledStatement, NULL);

if(return_value != SQLITE_OK)
{
throw SOMException("Error compiling statement\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Statement::Bind(int64_t inputPosition, int64_t inputValue)
{
if(sqlite3_bind_int64(compiledStatement,inputPosition, inputValue) != SQLITE_OK)
{
throw SOMException("Error binding int64 to statement at position " + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Statement::Bind(int64_t inputPosition, double inputValue)
{
if(sqlite3_bind_double(compiledStatement,inputPosition, inputValue) != SQLITE_OK)
{
throw SOMException("Error binding double to statement at position " + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Statement::Bind(int64_t inputPosition, bool inputValue)
{
if(sqlite3_bind_int64(compiledStatement,inputPosition, (int64_t) inputValue) != SQLITE_OK)
{
throw SOMException("Error binding bool to statement at position " + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This statement binds a value to a particular position in the compiled statement.  This particular instance binds text.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Statement::Bind(int64_t inputPosition, const std::string& inputValue)
{
if(sqlite3_bind_text64(compiledStatement,inputPosition, inputValue.c_str(), inputValue.size(), SQLITE_TRANSIENT, SQLITE_UTF8) != SQLITE_OK)
{
throw SOMException("Error binding string to statement at position " + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This statement binds a value to a particular position in the compiled statement.
@param inputPosition: The position to set to the given value
@param inputValue: The value to assign
@throws: This function throws an exception if the operation fails
*/
void Statement::BindBlob(int64_t inputPosition, const std::string& inputValue)
{
if(sqlite3_bind_blob64(compiledStatement,inputPosition, inputValue.c_str(), inputValue.size(), SQLITE_TRANSIENT) != SQLITE_OK)
{
throw SOMException("Error binding string to statement at position " + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This statement binds null to a particular position in the compiled statement.
@param inputPosition: The position to set to null
@throws: This function throws an exception if the operation fails
*/
void Statement::Bind(int64_t inputPosition)
{
if(sqlite3_bind_null(compiledStatement,inputPosition) != SQLITE_OK)
{
throw SOMException("Error binding null to statement at position " + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This function returns true if the stepped query has more rows left to return.
@return: True if there is more data to retrieve
*/
bool Statement::ResultAvailable() const
{
return ResultIsAvailable;
}

/**
This function retrieves the value stored from an executed query at the given position.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Statement::Retrieve(int64_t inputPosition, int64_t& inputResultBuffer)
{
int columnType = sqlite3_column_type(compiledStatement, inputPosition);

if(columnType == SQLITE_NULL)
{
return false;
}

if(columnType != SQLITE_INTEGER)
{
throw SOMException("Error retrieving int64 from statement at position" + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}

inputResultBuffer = sqlite3_column_int(compiledStatement, inputPosition);
return true;
}

/**
This function retrieves the value stored from an executed query at the given position.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Statement::Retrieve(int64_t inputPosition, double& inputResultBuffer)
{
int columnType = sqlite3_column_type(compiledStatement, inputPosition);

if(columnType == SQLITE_NULL)
{
return false;
}

if(columnType != SQLITE_FLOAT)
{
throw SOMException("Error retrieving double from statement at position" + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}

inputResultBuffer = sqlite3_column_double(compiledStatement, inputPosition);
return true;
}

/**
This function retrieves the value stored from an executed query at the given position.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Statement::Retrieve(int64_t inputPosition, bool& inputResultBuffer)
{
int64_t buffer = 0;
bool notNull = Retrieve(inputPosition, buffer);
if(!notNull)
{
return false;
}

inputResultBuffer = (bool) buffer;
return true;
}

/**
This function retrieves the value stored from an executed query at the given position.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Statement::Retrieve(int64_t inputPosition, std::string& inputResultBuffer)
{
int columnType = sqlite3_column_type(compiledStatement, inputPosition);

if(columnType == SQLITE_NULL)
{
return false;
}

if(columnType != SQLITE_TEXT)
{
throw SOMException("Error retrieving text from statement at position" + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}

const void* blobPointer = sqlite3_column_blob(compiledStatement, inputPosition);

inputResultBuffer = std::string((char*) blobPointer, sqlite3_column_bytes(compiledStatement, inputPosition));

return true;
}

/**
This function retrieves the value stored from an executed query at the given position.  This particular instance retrieve a blob.
@param inputPosition: The position to get the value from
@param inputResultBuffer: Where to store the result
@return: False if the retrieved value was null
@throws: This function throws an exception if the operation fails
*/
bool Statement::RetrieveBlob(int64_t inputPosition, std::string& inputResultBuffer)
{
int columnType = sqlite3_column_type(compiledStatement, inputPosition);

if(columnType == SQLITE_NULL)
{
return false;
}

if((columnType != SQLITE_BLOB) && (columnType != SQLITE_TEXT)) //Sqlite in returning text type from a blob column
{
throw SOMException("Error retrieving blob from statement at position" + std::to_string(inputPosition) + "\n", SQLITE3_ERROR, __FILE__, __LINE__);
}

const void* blobPointer = sqlite3_column_blob(compiledStatement, inputPosition);

inputResultBuffer = std::string((char*) blobPointer, sqlite3_column_bytes(compiledStatement, inputPosition));

return true;
}

/**
This function steps the statement to get the data or perform the operation.
@throws: This function throws an exception if the operation fails
*/
void Statement::Step()
{
int stepReturnValue = sqlite3_step(compiledStatement);

if(stepReturnValue == SQLITE_ROW)
{
ResultIsAvailable = true;
}
else if(stepReturnValue == SQLITE_DONE)
{
ResultIsAvailable = false;
}
else
{
throw SOMException("Error stepping statement\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This function resets the statement so that the operation can be done with new parameters.
@throws: This function throws an exception if the operation fails
*/
void Statement::Reset()
{
bool resetOK = sqlite3_reset(compiledStatement) == SQLITE_OK;

if(!resetOK)
{
throw SOMException("Error resetting statement\n", SQLITE3_ERROR, __FILE__, __LINE__);
}
}

/**
This function steps the statement and then resets it
@throws: This function throws an exception if the operations fail
*/
void Statement::StepAndReset()
{
Step();
Reset();
}

/**
This function calls finalize on the associated compiled statement.
*/
Statement::~Statement()
{
bool finalizedOK = sqlite3_finalize(compiledStatement) == SQLITE_OK;
assert(finalizedOK);
}

/**
This function calls "BEGIN TRANSACTION;" on the connection when it is created and "END TRANSACTION;" when it is destroyed.
@param inputDatabaseConnection: The connection to the database to use.
*/
TransactionGuard::TransactionGuard(DatabaseConnection& inputDatabaseConnection) : connection(&inputDatabaseConnection)
{
Statement transactionStatement("BEGIN TRANSACTION;", *connection);
transactionStatement.Step();
}

/**
Calls "END TRANSACTION;"
*/
TransactionGuard::~TransactionGuard()
{
try
{
Statement transactionStatement("END TRANSACTION;", *connection);
transactionStatement.Step();
}
catch(const std::exception &inputException)
{
assert(false);
}
}

}
}
