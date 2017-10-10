#pragma once

namespace GoodBot
{

/**
 * @brief The ExponentialMovingAverage is a simple object to make it easy to compute an exponential moving average.
 */
class ExponentialMovingAverage
{
public:
    /**
     * @brief This constructor sets the average to 0.0 and then re-writes it with the first value given to update (doing moving average after that).
     * @param newValueWeight : Ranges from (0.0 to 1.0).  At low numbers new values will be largely ignored, at high numbers it will have very little memory.
     */
    ExponentialMovingAverage(double newValueWeight);

    /**
     * @brief Create initial object with a default value for "prior" observations
     * @param initialValue: The "prior" value
     * @param newValueWeight : Ranges from (0.0 to 1.0).  At low numbers new values will be largely ignored, at high numbers it will have very little memory.
     */
    ExponentialMovingAverage(double initialValue, double newValueWeight);

    /**
     * @brief This function updates the moving average with a new value
     * @param newValue: The value to add to the moving average
     */
    void Update(double newValue);

    /**
     * @brief Get the value of the current average.
     * @return The exponential weighted moving average
     */
    double GetAverage() const;

private:
    bool SetAverageToNextValue;
    double Average;
    double NewValueWeight;
};
















}
