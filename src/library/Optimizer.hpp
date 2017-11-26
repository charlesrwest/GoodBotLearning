#pragma once

#include<algorithm>
#include<vector>
#include<functional>
#include<cmath>
#include<iostream>
#include "SOMException.hpp"

namespace GoodBot
{

template<class DataType>
class ValueRange
{
public:
    ValueRange(DataType min, DataType max);
    bool CanBeSplit() const;
    bool InRange(DataType value) const;
    DataType GetMin() const;
    DataType GetMax() const;
    bool operator==(const ValueRange<DataType>& rhs) const;

private:
    DataType Min;
    DataType Max;
};

template<class DataType>
bool CanBeSplit(const ValueRange<DataType>& valueRange);

template<class DataType>
bool CanBeSplit(const std::vector<ValueRange<DataType>>& ranges);

template<class DataType>
std::pair<ValueRange<DataType>, ValueRange<DataType>> Split(const ValueRange<DataType>& valueRange);

template<class DataType>
std::pair<std::vector<ValueRange<DataType>>, std::vector<ValueRange<DataType>>> Split(const std::vector<ValueRange<DataType>>& ranges);

template<class DataType>
std::pair<std::vector<ValueRange<DataType>>, std::vector<ValueRange<DataType>>> Split(const std::vector<ValueRange<DataType>>& ranges);

using IntegerRange = ValueRange<int64_t>;
using DoubleRange = ValueRange<double>;

template<class RandomnessGenerator>
int64_t Sample(const IntegerRange& range, RandomnessGenerator& randomness);

template<class RandomnessGenerator>
double Sample(const DoubleRange& range, RandomnessGenerator& randomness);

template<class DataType, class RandomnessGenerator>
std::vector<DataType> Sample(const std::vector<ValueRange<DataType>>& ranges, RandomnessGenerator& randomness);


class OptimizationEntry
{
public:
    OptimizationEntry(const std::vector<int64_t>& integerParameters, const std::vector<double>& doubleParameters, double value);
    double GetValue() const;
    const std::vector<int64_t>& GetIntegerParameters() const;
    const std::vector<double>& GetDoubleParameters() const;

protected:
    double Value;
    std::vector<int64_t> IntegerParameters;
    std::vector<double> DoubleParameters;
};

class OptimizationSpace
{
public:
    OptimizationSpace(int64_t depth, const std::vector<OptimizationEntry>& samples, const std::vector<IntegerRange>& integerRanges, const std::vector<DoubleRange>& doubleRanges);

    void AddSample(const OptimizationEntry& sample);
    bool SampleInSpace(const OptimizationEntry& sample) const;
    int64_t GetDepth() const;
    void IncrementDepth();
    double GetMinValue() const;
    const OptimizationEntry& GetMinSample() const;
    const std::vector<OptimizationEntry>& GetSamples() const;
    const std::vector<IntegerRange>& GetIntegerRanges() const;
    const std::vector<DoubleRange>& GetDoubleRanges() const;

protected:
    int64_t Depth;
    double MinValue;
    int64_t MinValueIndex = 0;

    std::vector<OptimizationEntry> Samples;
    std::vector<IntegerRange> IntegerRanges;
    std::vector<DoubleRange> DoubleRanges;
};

bool CanBeSplit(const OptimizationSpace& originalSpace);
std::vector<OptimizationSpace> Split(const OptimizationSpace& originalSpace);

double GetDiscountedValue(const OptimizationSpace& space, double discountRate);

template<class RandomnessGenerator>
void SampleInSpace(OptimizationSpace& region, std::function<double(const std::vector<double>&, const std::vector<int64_t>&)>& objective, RandomnessGenerator& randomness);

class Optimizer
{
  public:
    Optimizer(const std::function<double(const std::vector<double>&, const std::vector<int64_t>&)>& objectiveFunction,
              const std::vector<IntegerRange>& integerRanges, const std::vector<DoubleRange>& doubleRanges,
              const std::vector<OptimizationEntry>& priorSamples, int64_t samplesPerRegion, double discountRate);

    void StepSearch();
    void Search(int64_t searchDurationInMilliseconds);
    std::tuple<double, std::vector<int64_t>, std::vector<double>> GetBestParameters() const;

protected:
    std::mt19937 randomness;
    std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> ObjectiveFunction;
    std::vector<OptimizationSpace> SearchRegions;
    int64_t SamplesPerRegion;
    double DiscountRate;
};

template<class DataType>
ValueRange<DataType>::ValueRange(DataType min, DataType max) : Min(std::min(min, max)), Max(std::max(min, max))
{
}

template<class DataType>
bool ValueRange<DataType>::CanBeSplit() const
{
    return GetMin() != GetMax();
}

template<class DataType>
bool ValueRange<DataType>::InRange(DataType value) const
{
    return (GetMin() <= value) && (value <= GetMax());
}

template<class DataType>
DataType ValueRange<DataType>::GetMin() const
{
    return Min;
}

template<class DataType>
DataType ValueRange<DataType>::GetMax() const
{
    return Max;
}

template<class DataType>
bool ValueRange<DataType>::operator==(const ValueRange<DataType>& rhs) const
{
return (GetMax() == rhs.GetMax()) && (GetMin() == GetMin());
}

template<class DataType>
bool CanBeSplit(const ValueRange<DataType>& valueRange)
{
    return valueRange.GetMin() != valueRange.GetMax();
}

template<class DataType>
bool CanBeSplit(const std::vector<ValueRange<DataType>>& ranges)
{
    for(const ValueRange<DataType>& range : ranges)
    {
        if(!CanBeSplit(range))
        {
            return false;
        }
    }

    return true;
}

template<class DataType>
std::pair<ValueRange<DataType>, ValueRange<DataType>> Split(const ValueRange<DataType>& valueRange)
{
    DataType half_range = (valueRange.GetMax() - valueRange.GetMin())*.5;

    std::pair<ValueRange<DataType>, ValueRange<DataType>> result(ValueRange<DataType>(valueRange.GetMin(), valueRange.GetMin()+half_range), ValueRange<DataType>(valueRange.GetMin()+half_range, valueRange.GetMax()));

    return result;
}

template<class DataType>
std::pair<std::vector<ValueRange<DataType>>, std::vector<ValueRange<DataType>>> Split(const std::vector<ValueRange<DataType>>& ranges)
{
    std::pair<std::vector<ValueRange<DataType>>, std::vector<ValueRange<DataType>>> result;
    for(const ValueRange<DataType>& range : ranges)
    {
        std::pair<ValueRange<DataType>, ValueRange<DataType>> split_values = Split(range);
        result.first.emplace_back(split_values.first);
        result.second.emplace_back(split_values.second);
    }

    return result;
}

template<class RandomnessGenerator>
int64_t Sample(const IntegerRange& range, RandomnessGenerator& randomness)
{
    std::uniform_int_distribution<int64_t> dist(range.GetMin(), range.GetMax());

    return dist(randomness);
}

template<class RandomnessGenerator>
double Sample(const DoubleRange& range, RandomnessGenerator& randomness)
{
    std::uniform_real_distribution<double> dist(range.GetMin(), range.GetMax());

    return dist(randomness);
}

template<class DataType, class RandomnessGenerator>
std::vector<DataType> Sample(const std::vector<ValueRange<DataType>>& ranges, RandomnessGenerator& randomness)
{
    std::vector<DataType> result;
    for(const ValueRange<DataType>& range : ranges)
    {
        result.emplace_back(Sample(range, randomness));
    }

    return result;
}

template<class RandomnessGenerator>
void SampleInSpace(OptimizationSpace& region, std::function<double(const std::vector<double>&, const std::vector<int64_t>&)>& objective, RandomnessGenerator& randomness)
{
    try
    {
        std::vector<double> real_parameters = Sample(region.GetDoubleRanges(), randomness);
        std::vector<int64_t> integer_parameters = Sample(region.GetIntegerRanges(), randomness);

        double value = objective(real_parameters, integer_parameters);

        region.AddSample(OptimizationEntry(integer_parameters, real_parameters, value));
    }
    catch(const std::exception& exception)
    {
        std::cout << "Got an exception, supressing" << std::endl;
        region.AddSample(OptimizationEntry(integer_parameters, real_parameters, 1e15)); //Add large value to suppress sampling, but don't log to database.
    }
}

}
