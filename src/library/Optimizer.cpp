#include "Optimizer.hpp"
#include<chrono>

using namespace GoodBot;



OptimizationEntry::OptimizationEntry(const std::vector<int64_t>& integerParameters, const std::vector<double>& doubleParameters, double value) :
    Value(value), IntegerParameters(integerParameters), DoubleParameters(doubleParameters)
{
}

double OptimizationEntry::GetValue() const
{
    return Value;
}

const std::vector<int64_t>& OptimizationEntry::GetIntegerParameters() const
{
    return IntegerParameters;
}

const std::vector<double>& OptimizationEntry::GetDoubleParameters() const
{
    return DoubleParameters;
}

OptimizationSpace::OptimizationSpace(int64_t depth, const std::vector<OptimizationEntry>& samples, const std::vector<IntegerRange>& integerRanges, const std::vector<DoubleRange>& doubleRanges) : Depth(depth), IntegerRanges(integerRanges), DoubleRanges(doubleRanges)
{
    MinValue = std::numeric_limits<double>::max();

    for(const OptimizationEntry& entry : samples)
    {
        AddSample(entry);
    }
}

void OptimizationSpace::AddSample(const OptimizationEntry& sample)
{
    int64_t current_sample_index = GetSamples().size();
    Samples.emplace_back(sample);
    if(sample.GetValue() < GetMinValue())
    {
        MinValue = std::min(sample.GetValue(), MinValue);
        MinValueIndex = current_sample_index;
    }
}

bool OptimizationSpace::SampleInSpace(const OptimizationEntry& sample) const
{
    SOM_ASSERT(sample.GetIntegerParameters().size() == GetIntegerRanges().size(), "Invalid sample size");
    SOM_ASSERT(sample.GetDoubleParameters().size() == GetDoubleRanges().size(), "Invalid sample size");

    for(int64_t range_index = 0; range_index < IntegerRanges.size(); range_index++)
    {
        if(!IntegerRanges[range_index].InRange(sample.GetIntegerParameters()[range_index]))
        {
            return false;
        }
    }

    for(int64_t range_index = 0; range_index < DoubleRanges.size(); range_index++)
    {
        if(!DoubleRanges[range_index].InRange(sample.GetDoubleParameters()[range_index]))
        {
            return false;
        }
    }

    return true;
}

int64_t OptimizationSpace::GetDepth() const
{
    return Depth;
}

void OptimizationSpace::IncrementDepth()
{
    Depth++;
}

double OptimizationSpace::GetMinValue() const
{
    return MinValue;
}

const OptimizationEntry& OptimizationSpace::GetMinSample() const
{
    SOM_ASSERT(GetSamples().size() > 0, "No samples");
    return GetSamples()[MinValueIndex];
}

const std::vector<OptimizationEntry>& OptimizationSpace::GetSamples() const
{
    return Samples;
}

const std::vector<IntegerRange>& OptimizationSpace::GetIntegerRanges() const
{
    return IntegerRanges;
}

const std::vector<DoubleRange>& OptimizationSpace::GetDoubleRanges() const
{
    return DoubleRanges;
}

bool GoodBot::CanBeSplit(const OptimizationSpace& originalSpace)
{
    return CanBeSplit(originalSpace.GetIntegerRanges()) && CanBeSplit(originalSpace.GetDoubleRanges());
}

std::vector<OptimizationSpace> GoodBot::Split(const OptimizationSpace& originalSpace)
{
    if(!CanBeSplit(originalSpace))
    {
        return {};
    }

int64_t number_of_parameters = originalSpace.GetIntegerRanges().size() + originalSpace.GetDoubleRanges().size();
int64_t number_of_patterns = 0;

if(number_of_parameters > 0)
{
number_of_patterns = (1 << number_of_parameters);
}

std::vector<IntegerRange> left_integer_ranges;
std::vector<IntegerRange> right_integer_ranges;

std::tie(left_integer_ranges, right_integer_ranges) = Split(originalSpace.GetIntegerRanges());

std::vector<DoubleRange> left_double_ranges;
std::vector<DoubleRange> right_double_ranges;

std::tie(left_double_ranges, right_double_ranges) = Split(originalSpace.GetDoubleRanges());

//Make all possible spacial regions resulting from splitting each of the dimensions in half
std::vector<OptimizationSpace> results;
for(int64_t pattern = 0; pattern < number_of_patterns; pattern++)
{
std::vector<IntegerRange> integer_ranges;
std::vector<DoubleRange> double_ranges;

bool skip_pattern = false;

for(int64_t integer_range_index = 0; integer_range_index < originalSpace.GetIntegerRanges().size(); integer_range_index++)
{
    if(pattern & (1 << integer_range_index))
    {
        if(left_integer_ranges[integer_range_index] == right_integer_ranges[integer_range_index])
        {
                //This dimension couldn't split, so skip half6
                skip_pattern = true;
                break;
        }
        integer_ranges.emplace_back(left_integer_ranges[integer_range_index]);
    }
    else
    {
        integer_ranges.emplace_back(right_integer_ranges[integer_range_index]);
    }
}

if(skip_pattern)
{
        continue;
}

for(int64_t double_range_index = 0; double_range_index < originalSpace.GetDoubleRanges().size(); double_range_index++)
{
    if(pattern & (1 << (double_range_index + originalSpace.GetIntegerRanges().size())))
    {
        double_ranges.emplace_back(left_double_ranges[double_range_index]);
    }
    else
    {
        double_ranges.emplace_back(right_double_ranges[double_range_index]);
    }
}

results.emplace_back(originalSpace.GetDepth()+1, std::vector<OptimizationEntry>{}, integer_ranges, double_ranges);
}

//Could probably be made more efficient
for(const OptimizationEntry& entry : originalSpace.GetSamples())
{
    for(OptimizationSpace& region : results)
    {
        if(region.SampleInSpace(entry))
        {
            region.AddSample(entry);
        }
    }
}

return results;
}

double GoodBot::GetDiscountedValue(const OptimizationSpace& space, double discountRate)
{
    return space.GetMinValue()*std::pow(discountRate, space.GetDepth());
}

Optimizer::Optimizer(const std::function<double(const std::vector<double>&, const std::vector<int64_t>&)>& objectiveFunction,
          const std::vector<IntegerRange>& integerRanges, const std::vector<DoubleRange>& doubleRanges,
          const std::vector<OptimizationEntry>& priorSamples, int64_t samplesPerRegion, double discountRate) : ObjectiveFunction(objectiveFunction), SamplesPerRegion(samplesPerRegion), DiscountRate(discountRate)
{
    std::random_device device;
    randomness = std::mt19937(device());

    SearchRegions.emplace_back(0, std::vector<OptimizationEntry>{}, integerRanges, doubleRanges);

    //Double check prior samples are in space
    for(const OptimizationEntry& entry : priorSamples)
    {
        if(SearchRegions.back().SampleInSpace(entry))
        {
            SearchRegions.back().AddSample(entry);
        }
    }
}

void Optimizer::StepSearch()
{
    std::function<bool(const OptimizationSpace&, const OptimizationSpace&)> comparison = [&](const OptimizationSpace& lhs, const OptimizationSpace& rhs)
    {
            double lhs_value = lhs.GetMinValue()*pow(1.0/DiscountRate, lhs.GetDepth());
            double rhs_value = rhs.GetMinValue()*pow(1.0/DiscountRate, rhs.GetDepth());

            return lhs_value > rhs_value;
    };

    while(SearchRegions[0].GetSamples().size() >= SamplesPerRegion)
    {
        //This one has been handled, split it and move on
        std::pop_heap(SearchRegions.begin(), SearchRegions.end(), comparison);

        OptimizationSpace region_to_split = SearchRegions.back();
        SearchRegions.pop_back();

        if(CanBeSplit(region_to_split))
        {
            std::vector<OptimizationSpace> split_regions = Split(region_to_split);
            SOM_ASSERT(split_regions.size() > 1, "Splitting didn't split");
            for(const OptimizationSpace& region : split_regions )
            {
                SearchRegions.emplace_back(region);
                std::push_heap(SearchRegions.begin(), SearchRegions.end(),comparison);
            }
        }
        else
        {
            //Workaround to handle combinations which cannot be split
            region_to_split.IncrementDepth();
            SearchRegions.emplace_back(region_to_split);
            std::push_heap(SearchRegions.begin(), SearchRegions.end(), comparison);
        }
    }

    SampleInSpace(SearchRegions[0], ObjectiveFunction, randomness);
}

void Optimizer::Search(int64_t searchDurationInMilliseconds)
{
    auto GetCurrentTime = []()
    {
      return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    };

    int64_t start_time = GetCurrentTime();

    while((GetCurrentTime() - start_time) < searchDurationInMilliseconds)
    {
        StepSearch();
    }
}

std::tuple<double, std::vector<int64_t>, std::vector<double>> Optimizer::GetBestParameters() const
{
    int64_t best_region_index = -1;
    double min_value = std::numeric_limits<double>::max();
    for(int64_t region_index = 0; region_index < SearchRegions.size(); region_index++)
    {
        double region_min = SearchRegions[region_index].GetMinValue();
        if(region_min < min_value)
        {
            min_value = region_min;
            best_region_index = region_index;
        }
    }

    const OptimizationEntry& entry = SearchRegions[best_region_index].GetMinSample();

    return std::tuple<double, std::vector<int64_t>, std::vector<double>>{entry.GetValue(), entry.GetIntegerParameters(), entry.GetDoubleParameters()};
}
