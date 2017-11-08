#pragma once

#include<string>

#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include<google/protobuf/text_format.h>
#include "SOMException.hpp"
#include "NetOp.hpp"
#include "NetSpace.hpp"
#include "NetConstruction.hpp"

const double PI = 3.141592653589793238463;
const float  PI_F = 3.14159265358979f;

template<class DataType1, class DataType2>
void PairedRandomShuffle(typename std::vector<DataType1>& inputData, typename  std::vector<DataType2>& expectedOutputData)
{
assert(inputData.size() == expectedOutputData.size());

//Fisher-Yates shuffle
for(typename std::vector<DataType1>::size_type index = 0; index < inputData.size(); index++)
{
typename std::vector<DataType1>::size_type elementToSwapWithIndex = index + (rand() % (inputData.size() - index));
std::swap(inputData[index], inputData[elementToSwapWithIndex]);
std::swap(expectedOutputData[index], expectedOutputData[elementToSwapWithIndex]);
}
};

//Add function to allow printing of network architectures
std::function<void(const google::protobuf::Message&)> print = [&](const google::protobuf::Message& inputMessage)
{
std::string buffer;

google::protobuf::TextFormat::PrintToString(inputMessage, &buffer);

std::cout << buffer<<std::endl;
};

template<typename ValueType>
class PseudoImage
{
public:

    PseudoImage(int64_t width, int64_t height, int64_t depth) : values(height*width*depth), Height(height), Width(width), Depth(depth)
    {
    }

    ValueType GetValue(int64_t widthIndex, int64_t heightIndex, int64_t depthIndex) const
    {
        return values[ToFlatIndex(widthIndex, heightIndex, depthIndex)];
    }

    ValueType SetValue(ValueType value, int64_t widthIndex, int64_t heightIndex, int64_t depthIndex)
    {
        values[ToFlatIndex(widthIndex, heightIndex, depthIndex)] = value;
    }

    int64_t GetWidth() const
    {
        return Width;
    }

    int64_t GetHeight() const
    {
        return Height;
    }

    int64_t GetDepth() const
    {
        return Depth;
    }

    const ValueType* GetData() const
    {
        return &(values[0]);
    }

    int64_t GetSize() const
    {
        return values.size();
    }

private:
    int64_t ToFlatIndex(int64_t widthIndex, int64_t heightIndex, int64_t depthIndex) const
    {
        int64_t flat_index = (depthIndex * GetHeight() + heightIndex) * GetWidth() + widthIndex;
        if(!((flat_index >= 0) && (flat_index < values.size())))
        {
            int64_t i = 5;
        }

        return flat_index;
    }

    int64_t Height;
    int64_t Width;
    int64_t Depth;

    std::vector<ValueType> values;
};

template<typename ValueType>
void Fill(ValueType value, PseudoImage<ValueType>& image)
{
    for(int64_t x = 0; x < image.GetWidth(); x++)
    {
        for(int64_t y = 0; y < image.GetHeight(); y++)
        {
            for(int64_t depth = 0; depth < image.GetDepth(); depth++)
            {
                image.SetValue(value, x, y, depth);
            }
         }
    }
}

template<typename ValueType>
void DrawCircle(int64_t circleX, int64_t circleY, double innerRadius, double outerRadius, ValueType fillValue, std::vector<int64_t> depths, PseudoImage<ValueType>& image)
{
    SOM_ASSERT(innerRadius >= 0, "Inner radius must be non-negative");
    SOM_ASSERT(outerRadius >= 0, "Outer radius must be non-negative");
    SOM_ASSERT(innerRadius < outerRadius, "Inner radius must be less than outer radius");

    //Inefficient but easy fill method -> scan every pixel and decide based on distance
    for(int64_t x = 0; x < image.GetWidth(); x++)
    {
        for(int64_t y = 0; y < image.GetHeight(); y++)
        {
            ValueType x_distance = x - circleX;
            ValueType y_distance = y - circleY;
            ValueType distance = sqrt(x_distance*x_distance + y_distance*y_distance);

            if((distance <= outerRadius) && (distance >= innerRadius))
            {
                for(int64_t depth : depths)
                {
                    image.SetValue(fillValue, x, y, depth);
                }
            }
        }
    }
}

template<typename ValueType>
void DrawSquare(int64_t centerX, int64_t centerY, int64_t innerDimension, int64_t outerDimension, ValueType fillValue, std::vector<int64_t> depths, PseudoImage<ValueType>& image)
{
    SOM_ASSERT(innerDimension >= 0, "Inner dimension must be non-negative");
    SOM_ASSERT(outerDimension >= 0, "Outer dimension must be non-negative");
    SOM_ASSERT(innerDimension < outerDimension, "Inner dimension must be less than outer dimension");

    int64_t x_outer_min = std::max<int64_t>((int64_t) (centerX-((outerDimension+.5)/2)), 0);
    int64_t y_outer_min = std::max<int64_t>((int64_t) (centerY-((outerDimension+.5)/2)), 0);
    int64_t x_outer_max = std::min<int64_t>((int64_t) (centerX+((outerDimension+.5)/2)), std::max<int64_t>(image.GetWidth()-1, 0));
    int64_t y_outer_max = std::min<int64_t>((int64_t) (centerY+((outerDimension+.5)/2)), std::max<int64_t>(image.GetHeight()-1, 0));

    int64_t x_inner_min = std::max<int64_t>((int64_t) (centerX-((innerDimension+.5)/2)), 0);
    int64_t y_inner_min = std::max<int64_t>((int64_t) (centerY-((innerDimension+.5)/2)), 0);
    int64_t x_inner_max = std::min<int64_t>((int64_t) (centerX+((innerDimension+.5)/2)), std::max<int64_t>(image.GetWidth()-1, 0));
    int64_t y_inner_max = std::min<int64_t>((int64_t) (centerY+((innerDimension+.5)/2)), std::max<int64_t>(image.GetHeight()-1, 0));

    for(int64_t x = x_outer_min; x <= x_outer_max; x++)
    {
        for(int64_t y = y_outer_min; y <= y_outer_max; y++)
        {
            if((y >= y_inner_min) && (y <= y_inner_max) && (x >= x_inner_min) && (x <= x_inner_max) )
            {
                continue;
            }

            for(int64_t depth : depths)
            {
                image.SetValue(fillValue, x, y, depth);
            }
        }
    }
}

template<typename ValueType>
void DrawImageAsAscii(const PseudoImage<ValueType>& image, int64_t depth, ValueType thresholdValue, char lessThanEqualValue, char greaterThanValue, std::ostream& output_stream)
{
    for(int64_t y = 0; y < image.GetHeight(); y++)
    {
        for(int64_t x = 0; x < image.GetWidth(); x++)
        {
            if(image.GetValue(x, y, depth) <= thresholdValue)
            {
                output_stream << lessThanEqualValue;
            }
            else
            {
                output_stream << greaterThanValue;
            }
        }
        output_stream << std::endl;
    }
}

template<typename ValueType>
std::pair<std::vector<int32_t>, std::vector<PseudoImage<ValueType>>> CreateShapeCategorizationImageTrainingData(ValueType defaultValue, ValueType shapeFillValue, int64_t imageDepth, const std::vector<int64_t>& depthsToShapeFill)
{
    std::pair<std::vector<int32_t>, std::vector<PseudoImage<ValueType>>> result;
    std::vector<int32_t>& labels = result.first;
    std::vector<PseudoImage<ValueType>>& images = result.second;

    for(int64_t x_offset = -3; x_offset <= 3; x_offset++ )
    {
        for(int64_t y_offset = -3; y_offset <= 3; y_offset++ )
        {
            //Add square example
            images.emplace_back(20, 20, imageDepth);
            PseudoImage<ValueType>& square_image = images.back();
            Fill<ValueType>(defaultValue, square_image);
            DrawSquare<ValueType>(10+x_offset, 10+y_offset, 8, 10, shapeFillValue, depthsToShapeFill, square_image);

            //Squares get label 0
            labels.emplace_back(0);

            //Add circle example
            images.emplace_back(20, 20, imageDepth);
            PseudoImage<ValueType>& circle_image = images.back();
            Fill<ValueType>(defaultValue, circle_image);
            DrawCircle<ValueType>(10+x_offset, 10+y_offset, 3.0, 5.0, shapeFillValue, depthsToShapeFill, circle_image);

            //Circles get label 1
            labels.emplace_back(1);
        }
    }

    return result;
}

template<typename ValueType>
std::pair<std::vector<std::array<float, 2>>, std::vector<PseudoImage<ValueType>>> CreateShape2DLocalizationImageTrainingData(ValueType defaultValue, ValueType shapeFillValue, int64_t imageDepth, const std::vector<int64_t>& depthsToShapeFill)
{
    std::pair<std::vector<std::array<float, 2>>, std::vector<PseudoImage<ValueType>>> result;
    std::vector<std::array<float, 2>>& labels = result.first;
    std::vector<PseudoImage<ValueType>>& images = result.second;
    int64_t image_dimension = 20;

    for(int64_t x_offset = -3; x_offset <= 3; x_offset++ )
    {
        for(int64_t y_offset = -3; y_offset <= 3; y_offset++ )
        {
            //Add square example
            images.emplace_back(image_dimension, image_dimension, imageDepth);
            PseudoImage<ValueType>& square_image = images.back();
            Fill<ValueType>(defaultValue, square_image);
            int64_t center_x = 10+x_offset;
            int64_t center_y = 10+y_offset;
            DrawSquare<ValueType>(center_x, center_y, 8, 10, shapeFillValue, depthsToShapeFill, square_image);

            //Move to approximately +-1.0
            float normalized_center_x = ((center_x)/((double) 2*image_dimension)) - 1;
            float normalized_center_y = ((center_y)/((double) 2*image_dimension)) - 1;

            //Store center of square after normalization
            labels.emplace_back(std::array<float, 2>{normalized_center_x, normalized_center_y});
        }
    }

    return result;
}

template<typename ValueType>
void VisualizeTrainingData(const std::vector<int32_t>& labels, const std::vector<PseudoImage<ValueType>>& images, ValueType threshold)
{
    SOM_ASSERT(labels.size() == images.size(), "Number of labels and images must match");

    //Draw all generated training data
    for(int64_t example_index = 0; example_index < labels.size(); example_index++)
    {
        std::cout << "Label: " << labels[example_index] << std::endl << std::endl;

        for(int64_t depth = 0; depth < images[example_index].GetDepth(); depth++)
        {
            DrawImageAsAscii<char>(images[example_index], depth, threshold, 'O', 'X', std::cout);

            std::cout << std::endl;
        }
    }
}

bool BlobNamesFound(const std::vector<std::string>& blobNames, const caffe2::Workspace& workspace);

bool BlobShapeMatches(const std::string& blobName, const std::vector<int64_t>& expectedShape, const caffe2::Workspace& workspace);

struct SimpleTestConstraintData
{
double a;
double b;
};

double SimpleTestObjectiveFunction(const std::vector<double>& input, std::vector<double>& gradient, void* userData);

double SimpleTestVConstraint(const std::vector<double>& input, std::vector<double>& gradient, void* userData);
