#include "ExponentialMovingAverage.hpp"

using namespace GoodBot;

/**
 * @brief This constructor sets the average to 0.0 and then re-writes it with the first value given to update (doing moving average after that).
 * @param newValueWeight : Ranges from (0.0 to 1.0).  At low numbers new values will be largely ignored, at high numbers it will have very little memory.
 */
ExponentialMovingAverage::ExponentialMovingAverage(double newValueWeight) : Average(0.0), NewValueWeight(newValueWeight), SetAverageToNextValue(true)
{

}

ExponentialMovingAverage::ExponentialMovingAverage(double initialValue, double newValueWeight) : Average(initialValue), NewValueWeight(newValueWeight), SetAverageToNextValue(false)
{
}

void ExponentialMovingAverage::Update(double newValue)
{
    if(SetAverageToNextValue)
    {
        Average = newValue;
        SetAverageToNextValue = false;
    }
    else
    {
        Average = Average*(1.0 - NewValueWeight) + newValue*NewValueWeight;
    }
}

double ExponentialMovingAverage::GetAverage() const
{
    return Average;
}
