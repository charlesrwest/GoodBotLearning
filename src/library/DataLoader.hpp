#pragma once

#include<cstdint>

namespace GoodBot
{



class DataLoader
{
  public:
    /**
     * This function reads one or more example blobs from a data source and loads them into the requested buffer.
     * @param inputBufferAddress: The buffer to load the input to the network to
     * @param expectedOutputBufferAddress: The buffer to load the expected output of the network to
     * @param numberOfBlobs: How many blobs to read from the data source
     * @return: True if the end of an epoc was reached (if that isn't relavent, always returns false).
     */
    virtual bool ReadBlobs(char* inputBufferAddress, char* expectedOutputBufferAddress, int64_t numberOfBlobs);

    /**
     * This function reads one example blobs from a data source and loads them into the requested buffer.
     * @param inputBufferAddress: The buffer to load the input to the network to
     * @param expectedOutputBufferAddress: The buffer to load the expected output of the network to
     * @return: True if the end of an epoc was reached (if that isn't relavent, always returns false).
     */
    virtual bool ReadBlob(char* inputBufferAddress, char* expectedOutputBufferAddress) = 0;

    /**
     * How many bytes a single example's input to the network is.
     * @return: Number of bytes in input to the network if the batch size is 1
     */
    virtual int64_t GetInputDataSize() const = 0;

    /**
     * How many bytes a single example's expected network output is.
     * @return: Number of bytes the network outputs to the network if the batch size is 1
     */
    virtual int64_t GetExpectedOutputDataSize() const = 0;

    virtual ~DataLoader()
    {
    }
};



























}
