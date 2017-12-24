
#include "ExponentialMovingAverage.hpp"
#include "NetConstruction.hpp"
#include "TestHelpers.hpp"
#include "MemoryDataLoader.hpp"
#include "ExperimentLogger.hpp"
#include "caffe2/core/context_gpu.h"
#include<chrono>
//#include "Experimenter.hpp"
#include "Optimizer.hpp"
#include "UtilityFunctions.hpp"
#include "RandomizedFileDataLoader.hpp"
#include "SequentialFileDataLoader.hpp"

int main(int argc, char **argv)
{
    //Make data sources and define image dimensions
    //Define dataset name
    int64_t input_depth = 1;
    int64_t image_dimension = 200;
    int64_t batch_size = 32;

    //Produce the training data
    CreateAndSaveShape2DLocalizationImageTrainingData<char>(0, 100, input_depth, image_dimension, {0}, "squareLocalization.blobber");


    int64_t example_input_size = input_depth*image_dimension*image_dimension*sizeof(char);
    int64_t example_output_size =  2*sizeof(float);
    GoodBot::SplitBlobberFile(.8,  example_input_size + example_output_size, batch_size,
                          "squareLocalization.blobber", "squareLocalizationTrain.blobber", "squareLocalizationTest.blobber");

    //Setup experiment logger
    GoodBot::RandomizedFileDataLoader training_data_source("squareLocalizationTrain.blobber", example_input_size, example_output_size, 50, 10);
    GoodBot::SequentialFileDataLoader test_data_source("squareLocalizationTest.blobber", example_input_size, example_output_size, 50);

    GoodBot::ExperimentLogger logger("ExperimentLog.db");

    //~ 78,008,000 combinations + infinity with learning rate
    std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> TestHyperParameter =
            [&training_data_source, &test_data_source, &image_dimension, &input_depth, &logger, &batch_size](const std::vector<double>& doubleParameters, const std::vector<int64_t>& integerParameters)
    {
        //We expect one integer parameter which indicates the image depth to use
        SOM_ASSERT(integerParameters.size() == 5, "Expected single integer parameter");
        SOM_ASSERT(doubleParameters.size() == 1, "Expected no double parameters");

        //doubles:
        double learning_rate = doubleParameters[0];
    
        //integers:
        int64_t number_of_relu_layers = integerParameters[0];
        int64_t nodes_per_relu_layer = integerParameters[1];
        int64_t number_of_convolutional_modules = integerParameters[2];
        int64_t number_of_filters_at_base_layer = integerParameters[3];
        int64_t stride_level = integerParameters[4];

        std::cout << "Parameters: " ;
        for(double value : doubleParameters)
        {
            std::cout << " " << std::to_string(value);
        }

        for(int64_t value : integerParameters)
        {
            std::cout << " " << std::to_string(value);
        }
        std::cout << std::endl;

        //void AddEntry(const LogEntry& entry);
        GoodBot::LogEntry entry;
        entry.ExperimentGroupName = "SquareLocalizationResnet";
        entry.DataSetName = "SquareLocalization_" + std::to_string(image_dimension);
        entry.InvestigationMethod = "CRWSplittingNLO";
        entry.DoubleHyperParameters["LearningRate"] = {learning_rate};
        entry.IntegerHyperParameters["NumberOfReluLayers"] = {number_of_relu_layers};
        entry.IntegerHyperParameters["NodesPerReluLayer"] = {nodes_per_relu_layer};
        entry.IntegerHyperParameters["NumberOfConvolutionalModules"] = {number_of_convolutional_modules};
        entry.IntegerHyperParameters["NumberOfFiltersAtBaseLayer"] = {number_of_filters_at_base_layer};
        entry.IntegerHyperParameters["StrideLevel"] = {stride_level};

        int64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        //Create the Caffe2 workspace/context
        caffe2::DeviceOption option;
        option.set_device_type(caffe2::CUDA);
        caffe2::CUDAContext cuda_context(option);

        caffe2::Workspace workspace;

        GoodBot::NetSpace netspace(workspace);

        /** Create inputs/outputs */

        //Batch size, channel depth, width/height
        GoodBot::AddConstantFillOp("shape_2d_localize_input", "input_blob", 0,  caffe2::TensorProto::INT8, {batch_size, input_depth, image_dimension, image_dimension}, {"INIT"}, false, caffe2::CPU, netspace);

        //Batch size, expected category
        GoodBot::AddConstantFillOp("shape_2d_localize_expected_output", "expected_output_blob", 0.0f,  caffe2::TensorProto::FLOAT, {batch_size, 2}, {"INIT"}, false, caffe2::CPU, netspace);

        //TODO: Make operators to move input/expected output to GPU
        GoodBot::AddCopyCPUToGPU("shape_2d_localize_input_blob_mover", "input_blob", "input_blob_gpu", {}, netspace);
        GoodBot::AddCopyCPUToGPU("shape_2d_localize_expected_output_blob_mover", "expected_output_blob", "expected_output_blob_gpu", {}, netspace);

        //Make the training network
        GoodBot::AddCastOp("shape_2d_localize_cast", "input_blob_gpu", "INT8", "input_blob_casted", "FLOAT", {}, netspace);
        AddScaleOp("shape_2d_localize_scale", "input_blob_casted", "input_blob_scaled", (1.0/128.0), {}, netspace);

        //Add batch normalization at input to automatically scale/shift input into proper range
        GoodBot::AddSpatialBNModule("shape_2d_localize_input_BN", "input_blob_scaled", "shape_2d_localize_input_BN", .1, .001, "NCHW", {"TRAIN"}, {"TEST"}, netspace);

        auto AddRes = [&netspace](const std::string& input, int64_t outputDepth, int64_t stride, int64_t resIndex) -> std::string
        {
            std::string res_name = "shape_2d_localize_res_" + std::to_string(resIndex);

            std::string sum_name = res_name + "_branch_sum";

            auto MakeConvName= [&](int64_t convIndex)
            {
                return res_name + "_conv_" + std::to_string(convIndex);
            };

            auto MakeConvBNName = [&](int64_t convIndex)
            {
                return res_name + "_BN_" + std::to_string(convIndex);
            };

            std::string network_head = input;

            AddConvModule(MakeConvName(0), network_head, MakeConvName(0), outputDepth, stride, 1, 3, "XavierFill", "ConstantFill", netspace);
            AddReluOp(MakeConvName(0)+"_relu", MakeConvName(0), MakeConvName(0), {}, netspace);
            AddSpatialBNModule(MakeConvBNName(0), MakeConvName(0), MakeConvBNName(0), .1, .001, "NCHW", {"TRAIN"}, {"TEST"}, netspace);
            network_head = MakeConvBNName(0);

            AddConvModule(MakeConvName(1), network_head, MakeConvName(1), outputDepth, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
            AddReluOp(MakeConvName(1)+"_relu", MakeConvName(1), MakeConvName(1), {}, netspace);
            AddSpatialBNModule(MakeConvBNName(1), MakeConvName(1), MakeConvBNName(1), .1, .001, "NCHW", {"TRAIN"}, {"TEST"}, netspace);
            network_head = MakeConvBNName(1);

            AddConvModule(MakeConvName(2), network_head, MakeConvName(2), outputDepth, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
            AddSpatialBNModule(MakeConvBNName(2), MakeConvName(2), MakeConvBNName(2), .1, .001, "NCHW", {"TRAIN"}, {"TEST"}, netspace);
            network_head = MakeConvBNName(2);

            //If the network depth has changed or the stide is not 1, then add a 1x1 conv
            std::string skip_output_name = input;
            std::vector<int64_t> input_shape = GetBlobShape(input, netspace);
            std::vector<int64_t> network_head_shape = GetBlobShape(network_head, netspace);

            SOM_ASSERT(input_shape.size() > 1, "Input blob is of dim <2?");
            if((input_shape[2] != outputDepth) || stride != 1)
            {
                skip_output_name = MakeConvName(0) + "_skip";
                AddConvModule(MakeConvName(0) + "_skip", input, skip_output_name, outputDepth, stride, 0, 1, "XavierFill", "ConstantFill", netspace);
            }
            std::vector<int64_t> skip_output_name_shape = GetBlobShape(skip_output_name, netspace);

            AddSumOp(sum_name, {skip_output_name, MakeConvBNName(2)}, sum_name, {}, netspace);

            AddReluOp(sum_name+"_relu", sum_name, sum_name, {}, netspace);

            return sum_name;
        };

        //Add convolution modules as indicated by hyper parameters
        int64_t conv_depth = number_of_filters_at_base_layer;
        std::string network_head = "shape_2d_localize_input_BN";
        for( int64_t conv_module_index = 0; conv_module_index < number_of_convolutional_modules; conv_module_index++)
        {
            network_head = AddRes(network_head, conv_depth, stride_level, conv_module_index);
            conv_depth = conv_depth*stride_level;
        }

        auto MakeFCReluName = [](int64_t moduleIndex)
        {
          return "shape_2d_localize_fc_relu_" +std::to_string(moduleIndex);
        };
        for(int64_t fc_module_index = 0; fc_module_index < number_of_relu_layers; fc_module_index++)
        {
            AddFullyConnectedModuleWithActivation(MakeFCReluName(fc_module_index), network_head, MakeFCReluName(fc_module_index), nodes_per_relu_layer, "Relu", "XavierFill", "ConstantFill", netspace);
            network_head = MakeFCReluName(fc_module_index);
        }

        //fc 2
        //softmax
        AddFullyConnectedModule("shape_2d_localize_fc_output", network_head, "shape_2d_localize_fc_output", 2, "XavierFill", "ConstantFill", netspace);

        //Make loss/loopback for gradient
        AddSquaredL2DistanceOp("shape_2d_localize_loss_l2_dist", "shape_2d_localize_fc_output", "expected_output_blob_gpu", "shape_2d_localize_loss_l2_dist", {"TRAIN", "TEST"}, netspace);
        AddAveragedLossOp("shape_2d_localize_avg_loss", "shape_2d_localize_loss_l2_dist", "shape_2d_localize_avg_loss", {"TRAIN", "TEST"}, netspace);
        AddNetworkGradientLoopBack("shape_2d_localize_gradient_loop_back", "shape_2d_localize_avg_loss", {"TRAIN", "TEST"}, netspace);

        //Add gradient ops
        AddGradientOperators("shape_2d_localize", {"TRAIN"}, netspace);

        //Add solver ops -> lower than .001 learning rate n
        GoodBot::AddAdamSolvers("shape_2d_localize", netspace, .9, .999, 1e-5, -learning_rate);

        //Make ops to move output and loss to CPU
        GoodBot::AddCopyGPUToCPU("shape_2d_localize_avg_loss_mover", "shape_2d_localize_avg_loss", "shape_2d_localize_avg_loss_cpu", {}, netspace);
        GoodBot::AddCopyGPUToCPU("shape_2d_localize_fc_output_mover", "shape_2d_localize_fc_output", "shape_2d_localize_fc_output_cpu", {}, netspace);

        //Initialize network
        caffe2::NetDef shape_2d_localize_init_def = GoodBot::GetNetwork("shape_2d_localize", "INIT", false, netspace);

        shape_2d_localize_init_def.mutable_device_option()->set_device_type(caffe2::CUDA); //Set type to CUDA for all ops which have not directly forced CPU

        caffe2::NetBase* shape_2d_localize_init_net = workspace.CreateNet(shape_2d_localize_init_def);
        shape_2d_localize_init_net->Run();

        SOM_ASSERT(BlobNamesFound({"input_blob", "expected_output_blob"}, workspace), "Missing blob names");
        SOM_ASSERT(BlobShapeMatches("input_blob", {batch_size, input_depth, image_dimension, image_dimension}, workspace), "Incorrect input shape");
        SOM_ASSERT(BlobShapeMatches("expected_output_blob", {batch_size, 2}, workspace), "Incorrect output shape");

        caffe2::TensorCPU& input_blob = GoodBot::GetMutableTensor("input_blob", workspace);
        caffe2::TensorCPU& expected_output_blob = GoodBot::GetMutableTensor("expected_output_blob", workspace);

        //Create training network
        caffe2::NetDef shape_2d_localize_train_def = GoodBot::GetNetwork("shape_2d_localize", "TRAIN", true, netspace);
        shape_2d_localize_train_def.mutable_device_option()->set_device_type(caffe2::CUDA); //Set type to CUDA for all ops which have not directly forced CPU
        caffe2::NetBase* shape_2d_localize_train_net = workspace.CreateNet(shape_2d_localize_train_def);

        print(shape_2d_localize_train_def);

        //Create test network
        caffe2::NetDef shape_2d_localize_test_def = GoodBot::GetNetwork("shape_2d_localize", "TEST", true, netspace);
        shape_2d_localize_test_def.mutable_device_option()->set_device_type(caffe2::CUDA); //Set type to CUDA for all ops which have not directly forced CPU
        caffe2::NetBase* shape_2d_localize_test_net = workspace.CreateNet(shape_2d_localize_test_def);

        //print(shape_2d_localize_test_def);

        GoodBot::ExponentialMovingAverage training_error_moving_average(1.0 / (10));
        GoodBot::ExponentialMovingAverage test_error_moving_average(1.0 / (10));

        int64_t train_epoc_index = 0;
        int64_t number_of_training_epocs = 100;
        double best_test_loss = std::numeric_limits<double>::max();
        double best_test_loss_moving_average = std::numeric_limits<double>::max();
        double final_test_loss = std::numeric_limits<double>::max();
        std::array<double, 2> xDims{std::numeric_limits<double>::max(),std::numeric_limits<double>::min()};
        std::array<double, 2> yDims{std::numeric_limits<double>::max(),std::numeric_limits<double>::min()};
        int64_t iteration = 0;
        int64_t test_epoc_example_count = 1;
        for(; train_epoc_index < number_of_training_epocs; iteration++)
        {
        //Load data into blobs
        bool train_epoc_finished = training_data_source.ReadBlobs((char *) input_blob.mutable_data<int8_t>(),
                                                                  (char *) expected_output_blob.mutable_data<float>(), batch_size);
        //Run network with loaded instance
        shape_2d_localize_train_net->Run();

        //Get loss exponentially weighted moving average
        caffe2::TensorCPU& loss = GoodBot::GetMutableTensor("shape_2d_localize_avg_loss_cpu", workspace);

        caffe2::TensorCPU& shape_2d_localize_fc_output_cpu = GoodBot::GetMutableTensor("shape_2d_localize_fc_output_cpu", workspace);

        //std::cout << "Loss : " << loss.mutable_data<float>()[0] << std::endl;
        training_error_moving_average.Update(loss.mutable_data<float>()[0]);

        xDims[0] = std::min<double>(xDims[0], expected_output_blob.mutable_data<float>()[0]);
        xDims[1] = std::max<double>(xDims[1], expected_output_blob.mutable_data<float>()[0]);

        yDims[0] = std::min<double>(yDims[0], expected_output_blob.mutable_data<float>()[1]);
        yDims[1] = std::max<double>(yDims[1], expected_output_blob.mutable_data<float>()[1]);

        //std::cout << "Moving average loss ( " << iteration << " ): " << moving_average.GetAverage() << " " << expected_output_blob.mutable_data<float>()[0] << " " << expected_output_blob.mutable_data<float>()[1]
        //           << " " << shape_2d_localize_fc_output_cpu.mutable_data<float>()[0] << " " << shape_2d_localize_fc_output_cpu.mutable_data<float>()[1] << " " << train_epoc_index << std::endl;

        for(int64_t batch_index = 0; batch_index < batch_size; batch_index++)
        {
            //std::cout << "Expected/Received output: (" << expected_output_blob.mutable_data<float>()[batch_index*2] << ", " << expected_output_blob.mutable_data<float>()[batch_index*2 + 1] << ") (" << shape_2d_localize_fc_output_cpu.mutable_data<float>()[batch_index*2] << ", " << shape_2d_localize_fc_output_cpu.mutable_data<float>()[batch_index*2 + 1] << ")" << std::endl;
        }

        if(train_epoc_finished)
        {
            //Get the average test error
            bool last_example_in_test_epoc = false;
            test_epoc_example_count = 0;
            double average_test_loss = 0.0;
                            test_epoc_example_count = 0;
            while(!last_example_in_test_epoc)
            {
                last_example_in_test_epoc = test_data_source.ReadBlobs((char *) input_blob.mutable_data<int8_t>(),
                                                                              (char *) expected_output_blob.mutable_data<float>(), batch_size);

                //Run network with loaded instance
                shape_2d_localize_test_net->Run();

                for(int64_t batch_index = 0; batch_index < batch_size; batch_index++)
                {
                   //std::cout << "Expected/Received output: (" << expected_output_blob.mutable_data<float>()[batch_index*2] << ", " << expected_output_blob.mutable_data<float>()[batch_index*2 + 1] << ") (" << shape_2d_localize_fc_output_cpu.mutable_data<float>()[batch_index*2] << ", " << shape_2d_localize_fc_output_cpu.mutable_data<float>()[batch_index*2 + 1] << ")" << std::endl;
                }

                average_test_loss += loss.mutable_data<float>()[0];

                //std::cout << "Loss " << "(" << test_epoc_example_count << "): " << loss.mutable_data<float>()[0] << std::endl;

                test_epoc_example_count++;
            }
            average_test_loss = average_test_loss / test_epoc_example_count;
            test_error_moving_average.Update(average_test_loss);

            std::cout << "test loss (" << train_epoc_index << "): " << average_test_loss << std::endl;
            std::cout << "Moving average training loss ( " << iteration << " ): " << training_error_moving_average.GetAverage() << std::endl;
            final_test_loss = average_test_loss;
            best_test_loss = std::min(average_test_loss, best_test_loss);
            best_test_loss_moving_average = std::min(best_test_loss_moving_average, test_error_moving_average.GetAverage());

            if(test_error_moving_average.GetAverage() > (2.0*best_test_loss_moving_average))
            {
                //Test error is starting to go up, so early exit
                break;
            }

            train_epoc_index++;
        }
        }

        //std::cout << "X dims: (" << xDims[0] << ", " << xDims[1] << ")" << std::endl;
        //std::cout << "Y dims: (" << yDims[0] << ", " << yDims[1] << ")" << std::endl;

        int64_t finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        //Finish logging
        entry.BestTestError = best_test_loss;
        entry.FinalTestError = final_test_loss;
        entry.TimeOfCompletion = finish_time;
        entry.RunTime = finish_time - start_time;
        entry.NumberOfTrainingIterations = iteration;
        entry.NumberOfTrainingEpocs = train_epoc_index;

        //TODO: Make associated netspace functions
        entry.HashOfNetspace = "";
        entry.NetspaceSummary = "";
        entry.DoubleHyperParameters["TrainingLossMovingAverage"] = {training_error_moving_average.GetAverage()};
        entry.IntegerHyperParameters["BatchSize"] = {batch_size};

        logger.AddEntry(entry);
        std::cout << "best test loss: " << best_test_loss << std::endl;
        std::cout << "Moving average loss ( " << iteration << " ): " << training_error_moving_average.GetAverage() << std::endl;

        return best_test_loss;
    };


    try
    {
        TestHyperParameter({.0001}, {0, 660, 3, 10, 1});
    }
    catch(const std::exception& exception)
    {
        std::cout << "Got an exception (" << exception.what() <<  "), supressing" << std::endl;
    }


    //Pass in parameters:
    //doubles:
    //Learning rate: [.001, .00001]

    //integers:
    //#Fully connected rectified layers (after conv layers): [0,6]
    //# nodes in fully connected layers: [10, 2000]
    //#Convolution models: [0, 6]
    //# filters at base layer: [1, 200]
    //Stride level: [1,4]

    std::vector<GoodBot::IntegerRange> integer_ranges;
    std::vector<GoodBot::DoubleRange> double_ranges;
    integer_ranges.emplace_back(0, 0);
    integer_ranges.emplace_back(0, 1000);
    integer_ranges.emplace_back(0,4);
    integer_ranges.emplace_back(1,30);
    integer_ranges.emplace_back(1,2);

    double_ranges.emplace_back(0.001, .0005);

    int64_t MaxRunTimeMilliSeconds = 1000*60*60*24*7; //Go for about a week

    GoodBot::Optimizer experimenter(TestHyperParameter, integer_ranges, double_ranges,
    {}, 1000, .5);

    experimenter.Search(MaxRunTimeMilliSeconds);

    return 0;
}
