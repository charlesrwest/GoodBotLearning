#include "Experimenter.hpp"
#include<random>
#include<chrono>
#include<set>
#include<iostream>

using namespace GoodBot;

Investigator::Investigator(const InvestigatorRunConstraints& constraints,
             const std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> objectiveFunction) : Objective(objectiveFunction), Constraints(constraints)
{
}

template<class DataType>
std::size_t GenerateHash(const std::vector<DataType>& data)
{
    std::size_t hash = 0x0;

    for(const DataType& entry : data)
    {
        hash += std::hash<DataType>{}(entry);
    }

    return hash;
}

double Investigator::Optimize()
{
    //Don't have a good mixed integer optimization method right now, so gonna do random sampling

    //Make random parameter generators
    std::vector<std::uniform_int_distribution<int64_t>> int_arg_generators;
    std::vector<std::uniform_real_distribution<double>> double_arg_generators;

    for(const std::pair<int64_t, int64_t>& integer_constraint : Constraints.IntegerConstraints)
    {
        int_arg_generators.emplace_back(integer_constraint.first, integer_constraint.second);
    }

    for(const std::pair<double, double>& double_constraint : Constraints.DoubleConstraints)
    {
        double_arg_generators.emplace_back(double_constraint.first, double_constraint.second);
    }

    std::random_device randomness;
    std::mt19937 generator(randomness());

    auto GenerateIntegerArguments = [&]()
    {
        std::vector<int64_t> arguments;
        for(std::uniform_int_distribution<int64_t>& distribution : int_arg_generators)
        {
            arguments.emplace_back(distribution(generator));
        }
        return arguments;
    };

    auto GenerateDoubleArguments = [&]()
    {
        std::vector<double> arguments;
        for(std::uniform_real_distribution<double>& distribution : double_arg_generators)
        {
            arguments.emplace_back(distribution(generator));
        }
        return arguments;
    };

    auto GetCurrentTime = []()
    {
      return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    };
    int64_t start_time = GetCurrentTime();
    std::set<std::pair<std::size_t, std::size_t>> hashes; //Integer argument hash, double argument hash

    double smallest_value = std::numeric_limits<double>::max();
    while((GetCurrentTime() - start_time) < Constraints.MaxRunTimeMilliSeconds)
    {
        try
        {
        //Generate and try random parameters
        std::vector<int64_t> integer_parameters = GenerateIntegerArguments();
        std::vector<double> double_parameters = GenerateDoubleArguments();

        std::pair<std::size_t, std::size_t> hash{GenerateHash(integer_parameters), GenerateHash(double_parameters)};

        if(hashes.count(hash) > 0)
        {
            //Did this one already, so skip
            continue;
        }

        double value = Objective(double_parameters, integer_parameters);
        hashes.insert(hash);
        smallest_value = std::min(value, smallest_value);
        }
        catch(const std::exception& exception)
        {
            std::cout << "Got an exception, supressing" << std::endl;
        }
    }

    return smallest_value;
}
