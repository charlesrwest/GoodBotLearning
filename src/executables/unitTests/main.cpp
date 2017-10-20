#define CATCH_CONFIG_MAIN //Make main function automatically
#include "catch.hpp"
#include<cstdlib>
#include<string>

#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include<google/protobuf/text_format.h>
#include "FullyConnectedModuleDefinition.hpp"
#include<iostream>
#include<cmath>
#include<cassert>
#include "AveragedL2LossModuleDefinition.hpp"
#include "AdamSolver.hpp"
#include<fstream>
#include<random>
#include<cstdio>
#include "DataLoader.hpp"
#include "SOMScopeGuard.hpp"
#include "AveragedLossLayerDefinition.hpp"
#include "LabelCrossEntropyOperator.hpp"
#include "SoftMaxLayerDefinition.hpp"
#include "FullyConnectedOperator.hpp"
#include "SOMException.hpp"

#include "ExponentialMovingAverage.hpp"
#include "NetConstruction.hpp"

const double PI  =3.141592653589793238463;
const float  PI_F=3.14159265358979f;

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

static const uint8_t BLUE_OFFSET = 0;
static const uint8_t GREEN_OFFSET = 1;
static const uint8_t RED_OFFSET = 2;

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
std::pair<std::vector<int32_t>, std::vector<PseudoImage<ValueType>>> CreateShapeImageTrainingData(ValueType defaultValue, ValueType shapeFillValue, int64_t imageDepth, const std::vector<int64_t>& depthsToShapeFill)
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

bool BlobNamesFound(const std::vector<std::string>& blobNames, const caffe2::Workspace& workspace)
{
std::vector<std::string> current_blobs = workspace.Blobs();

for(const std::string& blob_name : blobNames)
{
if(std::find(current_blobs.begin(), current_blobs.end(), blob_name) == current_blobs.end())
{
return false;
}
}

return true;
}

bool BlobShapeMatches(const std::string& blobName, const std::vector<int64_t>& expectedShape, const caffe2::Workspace& workspace)
{
caffe2::TensorCPU tensor = GoodBot::GetTensor(*workspace.GetBlob(blobName));

return expectedShape == tensor.dims();
}

TEST_CASE("Draw shapes", "[Example]")
{
    std::vector<int32_t> labels;
    std::vector<PseudoImage<char>> images;

    std::tie(labels, images) = CreateShapeImageTrainingData<char>(0, 100, 1, {0});

    REQUIRE(labels.size() > 0);
    REQUIRE(labels.size() == images.size());

    //VisualizeTrainingData<char>(labels, images, 0);
}

TEST_CASE("Simple conv network", "[Example]")
{
    //Loop through different input depths
    for(int64_t input_depth : {1, 2, 3})
    {

    //Create the Caffe2 workspace/context
    caffe2::Workspace workspace;
    caffe2::CPUContext context;

    GoodBot::NetSpace netspace(workspace);

    /** Create inputs/outputs */

    //Batch size, channel depth, width/height
    GoodBot::AddConstantFillOp("init_interfaces_input", "input_blob", 0,  caffe2::TensorProto::INT8, {1, input_depth, 20, 20}, {"INIT"}, false, netspace);

    //Batch size, expected category
    GoodBot::AddConstantFillOp("init_interfaces_expected_output", "expected_output_blob", 0,  caffe2::TensorProto::INT32, {1, 1}, {"INIT"}, false, netspace);

    //Create input blobs
    caffe2::NetDef input_init_network = GoodBot::GetNetwork("init_interfaces", "INIT", false, netspace);
    caffe2::NetBase* init_interfaces_net = workspace.CreateNet(input_init_network);
    init_interfaces_net->Run();

    REQUIRE(BlobNamesFound({"input_blob", "expected_output_blob"}, workspace));
    REQUIRE(BlobShapeMatches("input_blob", {1, input_depth, 20, 20}, workspace));
    REQUIRE(BlobShapeMatches("expected_output_blob", {1, 1}, workspace));

    caffe2::TensorCPU& input_blob = GoodBot::GetMutableTensor("input_blob", workspace);
    caffe2::TensorCPU& expected_output_blob = GoodBot::GetMutableTensor("expected_output_blob", workspace);

    //Produce the training data
    std::vector<int32_t> labels;
    std::vector<PseudoImage<char>> images;

    //If we have a depth of more than one, make a copy of the training data with exactly one of the depth layers drawn on for each possible depth.
    for(int64_t fill_depth_index = 0; fill_depth_index < input_depth; fill_depth_index++)
    {
        std::vector<int32_t> labels_buffer;
        std::vector<PseudoImage<char>> images_buffer;

        std::tie(labels_buffer, images_buffer) = CreateShapeImageTrainingData<char>(0, 100, input_depth, {fill_depth_index});

        labels.insert(labels.end(), labels_buffer.begin(), labels_buffer.end());
        images.insert(images.end(), images_buffer.begin(), images_buffer.end());
    }


    PairedRandomShuffle(images, labels);

    //Make the training network
    GoodBot::AddCastOp("shape_class_cast", "input_blob", "INT8", "input_blob_casted", "FLOAT", {}, netspace);
    AddScaleOp("shape_class_scale", "input_blob_casted", "input_blob_scaled", (1.0/128.0), {}, netspace);

    //conv (3x3, 20 channels)
    //conv (3x3, 20 channels)
    //max pool (stride = 2) 20x20 -> 10x10
    //Relu
    AddConvModule("shape_class_conv_1", "input_blob_scaled", "shape_class_conv_1", 32, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
    AddConvModule("shape_class_conv_2", "shape_class_conv_1", "shape_class_conv_2", 32, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
    AddMaxPoolOp("shape_class_max_pool_1", "shape_class_conv_2", "shape_class_max_pool_1", 2, 0, 2, "NCHW", {}, netspace);
    AddReluOp("shape_class_max_pool_relu_1", "shape_class_max_pool_1", "shape_class_max_pool_1", {}, netspace);

    //conv (3x3, 20 channels)
    //conv (3x3, 20 channels)
    //max pool (stride = 2) 10x10 -> 5x5
    //Relu
    AddConvModule("shape_class_conv_3", "shape_class_max_pool_1", "shape_class_conv_3", 64, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
    AddConvModule("shape_class_conv_4", "shape_class_conv_3", "shape_class_conv_4", 64, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
    AddMaxPoolOp("shape_class_max_pool_2", "shape_class_conv_4", "shape_class_max_pool_2", 2, 0, 2, "NCHW", {}, netspace);
    AddReluOp("shape_class_max_pool_relu_2", "shape_class_max_pool_2", "shape_class_max_pool_2", {}, netspace);

    //relu fc 500
    //relu fc 500
    //fc 2
    //softmax
    AddFullyConnectedModuleWithActivation("shape_class_fc_1", "shape_class_max_pool_2", "shape_class_fc_1", 512, "Relu", "XavierFill", "ConstantFill", netspace);
    AddFullyConnectedModuleWithActivation("shape_class_fc_2", "shape_class_fc_1", "shape_class_fc_2", 512, "Relu", "XavierFill", "ConstantFill", netspace);
    AddFullyConnectedModule("shape_class_fc_3", "shape_class_fc_2", "shape_class_fc_3", 2, "XavierFill", "ConstantFill", netspace);
    AddSoftMaxOp("shape_class_softmax", "shape_class_fc_3", "shape_class_softmax", {}, netspace);

    //Make loss/loopback for gradient
    AddLabelCrossEntropyOp("shape_class_cross_entropy_loss", "shape_class_softmax", "expected_output_blob", "shape_class_cross_entropy_loss", {"TRAIN", "TEST"}, netspace);
    AddAveragedLossOp("shape_class_avg_loss", "shape_class_cross_entropy_loss", "shape_class_avg_loss", {"TRAIN", "TEST"}, netspace);
    AddNetworkGradientLoopBack("shape_class_gradient_loop_back", "shape_class_avg_loss", {"TRAIN", "TEST"}, netspace);

    //Add gradient ops
    AddGradientOperators("shape_class", {"TRAIN", "TEST"}, netspace);

    //Add solver ops
    GoodBot::AddAdamSolvers("shape_class", netspace);

    //Initialize network
    caffe2::NetDef shape_class_init_def = GoodBot::GetNetwork("shape_class", "INIT", false, netspace);
    caffe2::NetBase* shape_class_init_net = workspace.CreateNet(shape_class_init_def);
    shape_class_init_net->Run();

    //Create training network
    caffe2::NetDef shape_class_train_def = GoodBot::GetNetwork("shape_class", "TRAIN", true, netspace);
    caffe2::NetBase* shape_class_train_net = workspace.CreateNet(shape_class_train_def);

    GoodBot::ExponentialMovingAverage moving_average(1.0 / (10));

    int64_t number_of_training_iterations = 300;
    for(int64_t iteration = 0; iteration < number_of_training_iterations; iteration++)
    {
    //Shuffle data every epoc through
    if((iteration % labels.size()) == 0)
    {
    PairedRandomShuffle(images, labels);
    }

    //Load data into blobs
    SOM_ASSERT(images[iteration % images.size()].GetSize() == input_blob.nbytes(), "Image size differs from input_blob size (" + std::to_string(images[iteration % images.size()].GetSize()) + " vs " + std::to_string(input_blob.nbytes())) + ")";
    memcpy(input_blob.mutable_data<int8_t>(), images[iteration % images.size()].GetData(), input_blob.nbytes());

    SOM_ASSERT(sizeof(labels[iteration % labels.size()]) == expected_output_blob.nbytes(), "Label size mismatch");
    memcpy(expected_output_blob.mutable_data<int32_t>(), &labels[iteration % labels.size()], expected_output_blob.nbytes());

    //Run network with loaded instance
    shape_class_train_net->Run();

    //Get loss exponentially weighted moving average
    caffe2::TensorCPU& loss = GoodBot::GetMutableTensor("shape_class_avg_loss", workspace);

    moving_average.Update((double) *loss.mutable_data<float>());

    //std::cout << "Moving average loss ( " << iteration << " ): " << moving_average.GetAverage() << std::endl;
    }
    REQUIRE(moving_average.GetAverage() < .01);
    }
}

TEST_CASE("Try netop syntax", "[Example]")
{
GoodBot::Arg("hello", 1.0);
GoodBot::Arg("hello", 2);
GoodBot::Arg("hello", "Yello");
GoodBot::Arg("hello", {1.0, 2.0});
GoodBot::Arg("hello", {2, 3});
GoodBot::Arg("hello", {"Yello", "Mello"});

std::vector<GoodBot::Arg> arguments{{"hello", 1.0}, {"hello", 2}, {"hello", "Yello"}, {"hello", {1.0, 2.0}}, {"hello", {2, 3}}, {"hello", {"Yello", "Mello"}}};

GoodBot::CreateOpDef("Bob", {"input1", "input2"}, {"output1", "output2"}, "Type", {{"hello", 1.0}, {"hello", 2}, {"hello", "Yello"}, {"hello", {1.0, 2.0}}, {"hello", {2, 3}}, {"hello", {"Yello", "Mello"}}});

caffe2::Workspace workspace;
caffe2::CPUContext context;

GoodBot::NetSpace net_space(workspace);

std::string net_fill_name = "net1_constant_fill_op";
AddConstantFillOp(net_fill_name, "constant_fill_blob", ((float) 99.0), caffe2::TensorProto_DataType_FLOAT, {10, 100, 100}, {"INIT"}, true, net_space);

const GoodBot::NetOp& constant_fill_op = net_space.GetNetOp(net_fill_name);

REQUIRE(constant_fill_op.GetName() == net_fill_name);
REQUIRE(constant_fill_op.GetActiveModes() == std::vector<std::string>{"INIT"});
REQUIRE(constant_fill_op.IsTrainable() == true);
const caffe2::OperatorDef& constant_fill_op_def = constant_fill_op.GetOperatorDef();

REQUIRE(constant_fill_op_def.name() == net_fill_name);
REQUIRE(constant_fill_op_def.input_size() == 0);
REQUIRE(constant_fill_op_def.output_size() == 1);
REQUIRE(constant_fill_op_def.output(0) == "constant_fill_blob");
REQUIRE(constant_fill_op_def.type() == "ConstantFill");
REQUIRE(constant_fill_op_def.arg_size() == 3);

const caffe2::Argument& constant_fill_shape_arg = constant_fill_op_def.arg(0);
REQUIRE(constant_fill_shape_arg.name() == "shape");
REQUIRE(constant_fill_shape_arg.ints_size() == 3);
REQUIRE(constant_fill_shape_arg.ints(0) == 10);
REQUIRE(constant_fill_shape_arg.ints(1) == 100);
REQUIRE(constant_fill_shape_arg.ints(2) == 100);

const caffe2::Argument& constant_fill_value_arg = constant_fill_op_def.arg(1);
REQUIRE(constant_fill_value_arg.name() == "value");
REQUIRE(constant_fill_value_arg.f() == 99.0);

const caffe2::Argument& constant_fill_dtype_arg = constant_fill_op_def.arg(2);
REQUIRE(constant_fill_dtype_arg.name() == "dtype");
REQUIRE(constant_fill_dtype_arg.i() == caffe2::TensorProto_DataType_FLOAT);
}


//Returns <inputs, outputs>
std::pair<std::vector<float>, std::vector<float>> MakeSineApproximationTrainingData(int64_t numberOfTrainingExamples)
{
std::pair<std::vector<float>, std::vector<float>> results;
std::vector<float>& trainingInputs = results.first;
std::vector<float>& trainingExpectedOutputs = results.second;

for(int64_t trainingExampleIndex = 0; trainingExampleIndex < numberOfTrainingExamples; trainingExampleIndex++)
{
trainingInputs.emplace_back((((double) trainingExampleIndex)/(numberOfTrainingExamples+1))*2.0 - 1.0);
trainingExpectedOutputs.emplace_back(sin(((double) trainingExampleIndex)/(numberOfTrainingExamples+1)*2.0*PI));
}

return results;
}



//Make sure we can at least train a simple fully connected network
TEST_CASE("Test netspace sine approximation", "[Example]")
{
//Make 10000 training examples
int64_t numberOfTrainingExamples = 1000;
int64_t batch_size = 1;
REQUIRE((numberOfTrainingExamples % batch_size) == 0);

std::vector<float> trainingInputs;
std::vector<float> trainingExpectedOutputs;

std::tie(trainingInputs, trainingExpectedOutputs) = MakeSineApproximationTrainingData(numberOfTrainingExamples);

//Shuffle the examples
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);

//Create the Caffe2 workspace/context
caffe2::Workspace workspace;
caffe2::CPUContext context;

GoodBot::NetSpace netspace(workspace);

//Add operators to create input blobs
GoodBot::AddConstantFillOp("init_interfaces_input", "input_blob", 0.0f,  caffe2::TensorProto::FLOAT, {1, 1}, {"INIT"}, false, netspace);

GoodBot::AddConstantFillOp("init_interfaces_expected_output", "expected_output_blob", 0.0f,  caffe2::TensorProto::FLOAT, {1, 1}, {"INIT"}, false, netspace);

//Create input blobs
caffe2::NetDef input_init_network = GoodBot::GetNetwork("init_interfaces", "INIT", false, netspace);
caffe2::NetBase* init_interfaces_net = workspace.CreateNet(input_init_network);
init_interfaces_net->Run();

REQUIRE(BlobNamesFound({"input_blob", "expected_output_blob"}, workspace));
REQUIRE(BlobShapeMatches("input_blob", {1, 1}, workspace));
REQUIRE(BlobShapeMatches("expected_output_blob", {1, 1}, workspace));

caffe2::TensorCPU& input_blob = GoodBot::GetMutableTensor("input_blob", workspace);
caffe2::TensorCPU& expected_output_blob = GoodBot::GetMutableTensor("expected_output_blob", workspace);

//Make main net
AddFullyConnectedModuleWithActivation("sine_net_1", "input_blob", "sine_net_fc_1", 100, "Relu", "XavierFill", "ConstantFill", netspace);
AddFullyConnectedModuleWithActivation("sine_net_2", "sine_net_fc_1", "sine_net_fc_2", 100, "Relu", "XavierFill", "ConstantFill", netspace);
AddFullyConnectedModule("sine_net_3", "sine_net_fc_2", "output_blob", 1, "XavierFill", "ConstantFill", netspace);

//Make loss/loopback for gradient
AddSquaredL2DistanceOp("sine_net_loss_l2_dist", "output_blob", "expected_output_blob", "sine_net_loss_l2_dist", {"TRAIN", "TEST"}, netspace);
AddAveragedLossOp("sine_net_avg_loss", "sine_net_loss_l2_dist", "sine_net_avg_loss", {"TRAIN", "TEST"}, netspace);
AddNetworkGradientLoopBack("sine_net_gradient_loop_back", "sine_net_avg_loss", {"TRAIN", "TEST"}, netspace);

//Add gradient ops
AddGradientOperators("sine_net", {"TRAIN", "TEST"}, netspace);

//Add solver ops
GoodBot::AddAdamSolvers("sine_net", netspace);

//Initialize network
caffe2::NetDef sine_net_init_def = GoodBot::GetNetwork("sine_net", "INIT", false, netspace);

caffe2::NetBase* sine_net_init_net = workspace.CreateNet(sine_net_init_def);
sine_net_init_net->Run();

//Create training network
caffe2::NetDef sin_net_train_def = GoodBot::GetNetwork("sine_net", "TRAIN", true, netspace);
caffe2::NetBase* sine_net_train_net = workspace.CreateNet(sin_net_train_def);

//Train the network
int64_t numberOfTrainingIterations = 100000;

GoodBot::ExponentialMovingAverage moving_average(1.0 / (numberOfTrainingExamples / 2));

for(int64_t iteration = 0; iteration < numberOfTrainingIterations; iteration++)
{
//Shuffle data every epoc through
if((iteration % trainingInputs.size()) == 0)
{
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);
}

//Load data into network
memcpy(input_blob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()], input_blob.nbytes());
memcpy(expected_output_blob.mutable_data<float>(), &trainingExpectedOutputs[iteration % trainingExpectedOutputs.size()], expected_output_blob.nbytes());

//Run network in training mode
sine_net_train_net->Run();

caffe2::TensorCPU& loss = GoodBot::GetMutableTensor("sine_net_avg_loss", workspace);

moving_average.Update((double) *loss.mutable_data<float>());
}

REQUIRE(moving_average.GetAverage() < .01);
}

//Make sure we can handle a simple categorization network (learn XOR, categorizing into (0,1))
TEST_CASE("Test simple categorization network with netspace", "[Example]")
{
    //Make training examples
    int64_t numberOfTrainingExamples = 4;
    int64_t batch_size = 1;
    REQUIRE((numberOfTrainingExamples % batch_size) == 0);
    std::vector<std::array<float, 2>> trainingInputs;
    std::vector<int32_t> trainingExpectedOutputs;

    int64_t inputs_per_example = 2;
    int64_t outputs_per_example = 1;

    for(bool input_1_value : {false, true})
    {
    for(bool input_2_value : {false, true})
    {
    trainingInputs.emplace_back(std::array<float, 2>{(float) input_1_value, (float) input_2_value});
    trainingExpectedOutputs.emplace_back((int32_t) (input_1_value != input_2_value));
    }
    }

    //Shuffle the examples
    PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);

    //Create the Caffe2 workspace/context
    caffe2::Workspace workspace;
    caffe2::CPUContext context;

    GoodBot::NetSpace netspace(workspace);

    //Add operators to create input blobs
    GoodBot::AddConstantFillOp("init_interfaces_input", "input_blob", 0.0f,  caffe2::TensorProto::FLOAT, {1, 2}, {"INIT"}, false, netspace);
    GoodBot::AddConstantFillOp("init_interfaces_expected_output", "expected_output_blob", (int32_t) 0,  caffe2::TensorProto::INT32, {1, 1}, {"INIT"}, false, netspace);

    //Create input blobs
    caffe2::NetDef input_init_network = GoodBot::GetNetwork("init_interfaces", "INIT", false, netspace);
    caffe2::NetBase* init_interfaces_net = workspace.CreateNet(input_init_network);
    init_interfaces_net->Run();

    REQUIRE(BlobNamesFound({"input_blob", "expected_output_blob"}, workspace));
    REQUIRE(BlobShapeMatches("input_blob", {1, 2}, workspace));
    REQUIRE(BlobShapeMatches("expected_output_blob", {1, 1}, workspace));

    caffe2::TensorCPU& input_blob = GoodBot::GetMutableTensor("input_blob", workspace);
    caffe2::TensorCPU& expected_output_blob = GoodBot::GetMutableTensor("expected_output_blob", workspace);

    //Make main net
    AddFullyConnectedModuleWithActivation("xor_net_1", "input_blob", "xor_net_fc_1", 100, "Relu", "XavierFill", "ConstantFill", netspace);
    AddFullyConnectedModuleWithActivation("xor_net_2", "xor_net_fc_1", "xor_net_fc_2", 100, "Relu", "XavierFill", "ConstantFill", netspace);
    AddFullyConnectedModule("xor_net_3", "xor_net_fc_2", "xor_net_fc_3", 2, "XavierFill", "ConstantFill", netspace);
    AddSoftMaxOp("xor_net_softmax", "xor_net_fc_3", "xor_net_softmax", {}, netspace);

    //Make loss/loopback for gradient
    AddLabelCrossEntropyOp("xor_net_cross_entropy_loss", "xor_net_softmax", "expected_output_blob", "xor_net_cross_entropy_loss", {"TRAIN", "TEST"}, netspace);
    AddAveragedLossOp("xor_net_avg_loss", "xor_net_cross_entropy_loss", "xor_net_avg_loss", {"TRAIN", "TEST"}, netspace);
    AddNetworkGradientLoopBack("xor_net_gradient_loop_back", "xor_net_avg_loss", {"TRAIN", "TEST"}, netspace);

    //Add gradient ops
    AddGradientOperators("xor_net", {"TRAIN", "TEST"}, netspace);

    //Add solver ops
    GoodBot::AddAdamSolvers("xor_net", netspace);


    //Initialize network
    caffe2::NetDef xor_net_init_def = GoodBot::GetNetwork("xor_net", "INIT", false, netspace);
    caffe2::NetBase* xor_net_init_net = workspace.CreateNet(xor_net_init_def);
    xor_net_init_net->Run();

    //Create training network
    caffe2::NetDef xor_net_train_def = GoodBot::GetNetwork("xor_net", "TRAIN", true, netspace);
    caffe2::NetBase* xor_net_train_net = workspace.CreateNet(xor_net_train_def);

    GoodBot::ExponentialMovingAverage moving_average(1.0 / (100));

    int64_t number_of_training_iterations = 1000;
    for(int64_t iteration = 0; iteration < number_of_training_iterations; iteration++)
    {
    //Shuffle data every epoc through
    if((iteration % trainingInputs.size()) == 0)
    {
    PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);
    }

    //Load data into blobs
    memcpy(input_blob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()][0], input_blob.nbytes());
    memcpy(expected_output_blob.mutable_data<int32_t>(), &trainingExpectedOutputs[iteration % trainingExpectedOutputs.size()], expected_output_blob.nbytes());

    //Run network with loaded instance
    xor_net_train_net->Run();

    //Get loss exponentially weighted moving average
    caffe2::TensorCPU& loss = GoodBot::GetMutableTensor("xor_net_avg_loss", workspace);

    moving_average.Update((double) *loss.mutable_data<float>());
    }
    REQUIRE(moving_average.GetAverage() < .01);
}




//Make sure we can handle a simple categorization network (learn XOR, categorizing into (0,1))
TEST_CASE("Test simple categorization network", "[Example]")
{
//Make training examples
int64_t numberOfTrainingExamples = 4;
int64_t batch_size = 1;
REQUIRE((numberOfTrainingExamples % batch_size) == 0);
std::vector<std::array<float, 2>> trainingInputs;
std::vector<int32_t> trainingExpectedOutputs;

int64_t inputs_per_example = 2;
int64_t outputs_per_example = 1;

for(bool input_1_value : {false, true})
{
for(bool input_2_value : {false, true})
{
trainingInputs.emplace_back(std::array<float, 2>{(float) input_1_value, (float) input_2_value});
trainingExpectedOutputs.emplace_back((int32_t) (input_1_value != input_2_value));
}
}

//Shuffle the examples
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);

//Create the Caffe2 workspace/context
caffe2::Workspace workspace;
caffe2::CPUContext context;

//Create the blobs to inject the sine input/output for training
caffe2::TensorCPU& inputBlob = *workspace.CreateBlob("inputBlob")->GetMutable<caffe2::TensorCPU>();
inputBlob.Resize(batch_size, 2);
inputBlob.mutable_data<float>();

std::string trainingExpectedOutputBlobName = "trainingExpectedOutputBlobName";
caffe2::TensorCPU& expectedOutputBlob = *workspace.CreateBlob(trainingExpectedOutputBlobName)->GetMutable<caffe2::TensorCPU>();
expectedOutputBlob.Resize(batch_size, 1);
expectedOutputBlob.mutable_data<int32_t>();

//Define a 2 layer fully connected network, with softmax output and adam solver
GoodBot::CompositeComputeModuleDefinition network;
network.SetName("network");

network.AddModule(*(new GoodBot::FullyConnectedModuleDefinition("inputBlob", {100, 100}, network.Name() + "_fc_relu", 2, "XavierFill", "ConstantFill", "Relu")));

REQUIRE(network.modules.back()->GetOutputBlobNames().size() > 0);
network.AddModule(*(new GoodBot::FullyConnectedOperator(
network.Name() + "_fc1",
network.modules.back()->GetOutputBlobNames()[0],
network.Name() + "_fc1",
100,
2,
1
)));

REQUIRE(network.modules.back()->GetOutputBlobNames().size() > 0);
network.AddModule(*(new GoodBot::SoftMaxLayerDefinition(
{
network.modules.back()->GetOutputBlobNames()[0],
network.Name() + "_soft_max"
})));

REQUIRE(network.modules.back()->GetOutputBlobNames().size() > 0);
network.AddModule(*(new GoodBot::LabelCrossEntropyOperator(network.Name() + "_label_cross_entropy", network.Name() + "_soft_max", trainingExpectedOutputBlobName)));

network.AddModule(*(new GoodBot::AveragedLossLayerDefinition({network.Name() + "_label_cross_entropy", network.Name() + "_averaged_loss"})));

network.SetMode("TRAIN");

//Add a solver module for training/updating
GoodBot::AdamSolverParameters solverParams;
solverParams.moduleName = network.Name() + "adam_solver";
solverParams.trainableParameterNames = network.GetTrainableBlobNames();
solverParams.trainableParameterShapes = network.GetTrainableBlobShapes();

network.AddModule(*(new GoodBot::AdamSolver(solverParams)));

SECTION("Test training network", "[networkArchitecture]")
{
//Training the network, so set the mode to train
network.SetMode("TRAIN");

//Initialize the network by automatically generating the NetDef for network initialization in "TRAIN" mode
caffe2::NetDef trainingNetworkInitializationDefinition = network.GetInitializationNetwork();

//Create and run the initialization network.
caffe2::NetBase* initializationNetwork = workspace.CreateNet(trainingNetworkInitializationDefinition);
initializationNetwork->Run();

//Automatically generate the training network
caffe2::NetDef trainingNetworkDefinition = network.GetNetwork(workspace.Blobs());

//Instance the training network implementation
caffe2::NetBase* trainingNetwork = workspace.CreateNet(trainingNetworkDefinition);

//Create the deploy version of the network
network.SetMode("DEPLOY");

caffe2::NetDef deployNetworkDefinition = network.GetNetwork(workspace.Blobs());
caffe2::NetBase* deployNetwork = workspace.CreateNet(deployNetworkDefinition);

//Train the network
int64_t numberOfTrainingIterations = 1000;

for(int64_t iteration = 0; iteration < numberOfTrainingIterations; iteration++)
{
//Shuffle data every epoc through
if((iteration % trainingInputs.size()) == 0)
{
PairedRandomShuffle(trainingInputs, trainingExpectedOutputs);
}

//Load data into blobs
memcpy(inputBlob.mutable_data<float>(), &trainingInputs[iteration % trainingInputs.size()][0], inputBlob.nbytes());
memcpy(expectedOutputBlob.mutable_data<int32_t>(), &trainingExpectedOutputs[iteration % trainingExpectedOutputs.size()], expectedOutputBlob.nbytes());

//Run network with loaded instance
trainingNetwork->Run();

caffe2::TensorCPU& networkOutput = *workspace.GetBlob(network.Name() + "_soft_max")->GetMutable<caffe2::TensorCPU>();

caffe2::TensorCPU& network_loss = *workspace.GetBlob(network.Name() + "_averaged_loss")->GetMutable<caffe2::TensorCPU>();

//By half way through, the loss should be less than .01
if(iteration > (numberOfTrainingIterations / 2.0))
{
REQUIRE(network_loss.mutable_data<float>()[0] < 0.02);
}
}

}

}


TEST_CASE("Run iterator network and get data", "[Example]")
{
caffe2::Workspace workspace;
caffe2::CPUContext context;

std::string IteratorBlobName = "IterBlob";
caffe2::OperatorDef iterInitializationOperator;

iterInitializationOperator.set_type("ConstantFill");
iterInitializationOperator.add_output(IteratorBlobName);
caffe2::Argument& shape = *iterInitializationOperator.add_arg();
shape.set_name("shape");
shape.add_ints(1);
caffe2::Argument& value = *iterInitializationOperator.add_arg();
value.set_name("value");
value.set_i(0);
caffe2::Argument& dataType = *iterInitializationOperator.add_arg();
dataType.set_name("dtype");
dataType.set_i(caffe2::TensorProto_DataType_INT64); 

caffe2::NetDef initNetwork;
initNetwork.set_name("iter_init");

*initNetwork.add_op() = iterInitializationOperator;

caffe2::NetBase* initializationNetwork = workspace.CreateNet(initNetwork);
initializationNetwork->Run();

caffe2::OperatorDef iterOperator;

iterOperator.set_type("Iter");
iterOperator.add_input(IteratorBlobName);
iterOperator.add_output(IteratorBlobName);

caffe2::NetDef iterNetwork;
iterNetwork.set_name("iter_net");

*iterNetwork.add_op() = iterOperator;

caffe2::NetBase* iterationNetwork = workspace.CreateNet(iterNetwork);

caffe2::TensorCPU& iteratorBlob = *workspace.GetBlob(IteratorBlobName)->GetMutable<caffe2::TensorCPU>();

for(int64_t counter = 0; counter < 10000; counter++)
{
REQUIRE(iteratorBlob.mutable_data<int64_t>()[0] == counter);

iterationNetwork->Run();
}

}





//Returns sum of outputted sequence (can wrap around)
uint64_t WriteOutSequentialNumbers(uint64_t startValue, int64_t numberOfEntries, std::ofstream& outputStream)
{
uint64_t sum = 0;
for(int64_t entry_index = 0; entry_index < numberOfEntries; entry_index++)
{
uint64_t buffer = startValue + entry_index;
sum = sum + buffer;
outputStream.write((char*) &buffer, sizeof(uint64_t));
}

return sum;
} 

TEST_CASE("Write blobber file and pseudo-randomly read from it", "[Example]")
{
//Inputs are N sequential number starting at a random place and outputs are sequences of integers which start at the sum of the input sequence.
std::random_device randomness;
std::uniform_int_distribution<uint64_t> random_integer_generator(0, std::numeric_limits<uint64_t>::max());


auto TryCombination = [&](int64_t numberOfEntries, int64_t inputSequenceLength, int64_t outputSequenceLength, int64_t bufferSize, int64_t numberOfBuffers, int64_t maxRereadsBeforeRefill, int64_t numberOfTestReads, int64_t batchSize)
{
std::string temp_file_name = "temp.blobber";

//Delete the temporary file when we leave this function
SOMScopeGuard file_deleter([&](){ remove(temp_file_name.c_str()); });

int64_t input_blob_size = inputSequenceLength*sizeof(uint64_t);
int64_t output_blob_size = outputSequenceLength*sizeof(uint64_t);

//Write out the file
{
std::ofstream output_stream(temp_file_name,  std::ofstream::binary);

for(int64_t entry_index = 0; entry_index < numberOfEntries; entry_index++)
{
uint64_t output_sequence_start = 0;
uint64_t input_sequence_start = random_integer_generator(randomness);

//Write input sequence followed by output sequence
output_sequence_start = WriteOutSequentialNumbers(input_sequence_start, inputSequenceLength, output_stream);

WriteOutSequentialNumbers(output_sequence_start, outputSequenceLength, output_stream);
}
}

//Check file size
{
std::ifstream input_stream(temp_file_name,  std::ifstream::binary);

std::streampos start_position = input_stream.tellg();

input_stream.seekg(0, std::ifstream::end);

int64_t file_size = input_stream.tellg() - start_position;

REQUIRE(file_size == ((input_blob_size+output_blob_size)*numberOfEntries));
}

//Make a data loader and see if the retrieved blobs are coherent (match input/output rule)
GoodBot::DataLoader loader(temp_file_name, input_blob_size, output_blob_size, bufferSize, numberOfBuffers, maxRereadsBeforeRefill);

std::vector<uint64_t> input_buffer(inputSequenceLength*batchSize);
std::vector<uint64_t> output_buffer(outputSequenceLength*batchSize);

for(int64_t test_read_index = 0; test_read_index < numberOfTestReads; test_read_index++)
{
loader.ReadBlobs((char*) &input_buffer[0], (char*) &output_buffer[0], batchSize);

//Check each blob in batch
for(int64_t blob_index = 0; blob_index < batchSize; blob_index++)
{
int64_t input_sequence_offset = blob_index*inputSequenceLength;
uint64_t input_sum = input_buffer[input_sequence_offset];

//Check that input is a sequence
for(int64_t sequence_index = 1; sequence_index < inputSequenceLength; sequence_index++)
{
REQUIRE((input_buffer[sequence_index - 1 + input_sequence_offset]+1) == input_buffer[sequence_index + input_sequence_offset]);
input_sum += input_buffer[sequence_index + input_sequence_offset];
}

//Make sure output start matches
int64_t output_sequence_offset = blob_index*outputSequenceLength;
REQUIRE(output_buffer[output_sequence_offset] == input_sum);

//Check that output is a sequence
for(int64_t sequence_index = 1; sequence_index < outputSequenceLength; sequence_index++)
{
REQUIRE((output_buffer[sequence_index - 1 + output_sequence_offset]+1) == output_buffer[sequence_index + output_sequence_offset]);
}
}

}

};
 
//Write/read files with 1000 entries with input sequences of length 10 and output sequences of length 3, 5 buffers of 100 entries, with a max 2 rereads and 1000000 test reads, batch size 3
TryCombination(1000, 10, 3, 100, 5, 2, 1000000, 3);
}


