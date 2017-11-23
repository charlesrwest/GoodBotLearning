#pragma once

#include<vector>
#include<functional>
#include<tuple>
#include<nlopt.hpp>

namespace GoodBot
{

class InvestigatorRunConstraints
{
public:
std::vector<std::pair<double, double>> DoubleConstraints;
std::vector<std::pair<int64_t, int64_t>> IntegerConstraints;

int64_t MaxRunTimeMilliSeconds;
};

class Investigator
{
public:
Investigator(const InvestigatorRunConstraints& constraints,
             const std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> objectiveFunction);

double Optimize();

private:
InvestigatorRunConstraints Constraints;
std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> Objective;
};

}


