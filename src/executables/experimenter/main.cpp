
#include "ExponentialMovingAverage.hpp"
#include "NetConstruction.hpp"
#include "TestHelpers.hpp"
#include "MemoryDataLoader.hpp"
#include "ExperimentLogger.hpp"
#include "caffe2/core/context_gpu.h"
#include<chrono>
#include "Experimenter.hpp"

int main(int argc, char **argv)
{
    //Make data sources and define image dimensions
    //Define dataset name
    int64_t input_depth = 1;
    int64_t image_dimension = 30;

    //Produce the training data
    std::vector<std::array<float, 2>> training_expected_output;
    std::vector<PseudoImage<char>> images;

    //If we have a depth of more than one, make a copy of the training data with exactly one of the depth layers drawn on for each possible depth.
    for(int64_t fill_depth_index = 0; fill_depth_index < input_depth; fill_depth_index++)
    {
        std::vector<std::array<float, 2>> training_expected_output_buffer;
        std::vector<PseudoImage<char>> images_buffer;

        std::tie(training_expected_output_buffer, images_buffer) = CreateShape2DLocalizationImageTrainingData<char>(0, 100, input_depth, image_dimension, {fill_depth_index});

        training_expected_output.insert(training_expected_output.end(), training_expected_output_buffer.begin(), training_expected_output_buffer.end());
        images.insert(images.end(), images_buffer.begin(), images_buffer.end());
    }

    PairedRandomShuffle(images, training_expected_output);

    int64_t number_of_examples = images.size();

    std::cout << "There are " << number_of_examples << " examples with inputs of size " << images[0].GetSize() << " and outputs of size " << sizeof(float)*2 << std::endl;

    std::vector<char> network_inputs, network_training_inputs, network_test_inputs;
    AddDataToVector(images, network_inputs);

    std::vector<char> expected_network_outputs, expected_network_training_outputs, expected_network_test_outputs;
    AddDataToVector(training_expected_output, expected_network_outputs);

    double training_fraction = .8;

    std::tie(network_training_inputs, network_test_inputs) = SplitDataSet(training_fraction, network_inputs.size() / number_of_examples, network_inputs);

    std::tie(expected_network_training_outputs, expected_network_test_outputs) = SplitDataSet(training_fraction, expected_network_outputs.size() / number_of_examples, expected_network_outputs);

    int64_t number_of_training_examples = number_of_examples*training_fraction;
    int64_t number_of_test_examples = number_of_examples - number_of_training_examples;

    GoodBot::MemoryDataLoader training_data_source(number_of_training_examples, network_training_inputs, expected_network_training_outputs);
    GoodBot::MemoryDataLoader test_data_source(number_of_test_examples, network_test_inputs, expected_network_test_outputs);

    std::cout << "There are " << training_data_source.GetNumberOfExamples() << " training examples" << std::endl;
    std::cout << "There are " << test_data_source.GetNumberOfExamples() << " test examples" << std::endl;

    //Setup experiment logger


    GoodBot::ExperimentLogger logger("ExperimentLog.db");

    //~ 78,008,000 combinations + infinity with learning rate
    std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> TestHyperParameter =
            [&training_data_source, &test_data_source, &image_dimension, &input_depth, &logger](const std::vector<double>& doubleParameters, const std::vector<int64_t>& integerParameters)
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

        //void AddEntry(const LogEntry& entry);
        GoodBot::LogEntry entry;
        entry.ExperimentGroupName = "SquareLocalization";
        entry.DataSetName = "SquareLocalization_" + std::to_string(image_dimension);
        entry.InvestigationMethod = "Random";
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
        GoodBot::AddConstantFillOp("shape_2d_localize_input", "input_blob", 0,  caffe2::TensorProto::INT8, {1, input_depth, 20, 20}, {"INIT"}, false, caffe2::CPU, netspace);

        //Batch size, expected category
        GoodBot::AddConstantFillOp("shape_2d_localize_expected_output", "expected_output_blob", 0.0f,  caffe2::TensorProto::FLOAT, {1, 2}, {"INIT"}, false, caffe2::CPU, netspace);

        //TODO: Make operators to move input/expected output to GPU
        GoodBot::AddCopyCPUToGPU("shape_2d_localize_input_blob_mover", "input_blob", "input_blob_gpu", {}, netspace);
        GoodBot::AddCopyCPUToGPU("shape_2d_localize_expected_output_blob_mover", "expected_output_blob", "expected_output_blob_gpu", {}, netspace);

        //Make the training network
        GoodBot::AddCastOp("shape_2d_localize_cast", "input_blob_gpu", "INT8", "input_blob_casted", "FLOAT", {}, netspace);
        AddScaleOp("shape_2d_localize_scale", "input_blob_casted", "input_blob_scaled", (1.0/128.0), {}, netspace);

        //Add convolution modules as indicated by hyper parameters
        std::string network_head = "input_blob_scaled";
        int64_t conv_depth = number_of_filters_at_base_layer;
        auto MakeConvName = [](int64_t moduleIndex, int64_t subIndex)
        {
          if(subIndex < 2)
          {
              return "shape_2d_localize_conv_" + std::to_string(moduleIndex) + "_" + std::to_string(subIndex);
          }

          return "shape_2d_localize_conv_" +std::to_string(moduleIndex) + "_relu_" + std::to_string(subIndex);
        };
        for( int64_t conv_module_index = 0; conv_module_index < number_of_convolutional_modules; conv_module_index++)
        {
            AddConvModule(MakeConvName(conv_module_index, 0), network_head, MakeConvName(conv_module_index, 0), conv_depth, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
            AddConvModule(MakeConvName(conv_module_index, 1), MakeConvName(conv_module_index, 0), MakeConvName(conv_module_index, 1), conv_depth, stride_level, 1, 3, "XavierFill", "ConstantFill", netspace);
            AddReluOp(MakeConvName(conv_module_index, 2), MakeConvName(conv_module_index, 1), MakeConvName(conv_module_index, 1), {}, netspace);
            network_head = MakeConvName(conv_module_index, 1);
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
        AddGradientOperators("shape_2d_localize", {"TRAIN", "TEST"}, netspace);

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
        SOM_ASSERT(BlobShapeMatches("input_blob", {1, input_depth, 20, 20}, workspace), "Incorrect input shape");
        SOM_ASSERT(BlobShapeMatches("expected_output_blob", {1, 2}, workspace), "Incorrect output shape");

        caffe2::TensorCPU& input_blob = GoodBot::GetMutableTensor("input_blob", workspace);
        caffe2::TensorCPU& expected_output_blob = GoodBot::GetMutableTensor("expected_output_blob", workspace);


        //Create training network
        caffe2::NetDef shape_2d_localize_train_def = GoodBot::GetNetwork("shape_2d_localize", "TRAIN", true, netspace);
        shape_2d_localize_train_def.mutable_device_option()->set_device_type(caffe2::CUDA); //Set type to CUDA for all ops which have not directly forced CPU
        caffe2::NetBase* shape_2d_localize_train_net = workspace.CreateNet(shape_2d_localize_train_def);

        //Create test network
        caffe2::NetDef shape_2d_localize_test_def = GoodBot::GetNetwork("shape_2d_localize", "TEST", true, netspace);
        shape_2d_localize_test_def.mutable_device_option()->set_device_type(caffe2::CUDA); //Set type to CUDA for all ops which have not directly forced CPU
        caffe2::NetBase* shape_2d_localize_test_net = workspace.CreateNet(shape_2d_localize_test_def);

        GoodBot::ExponentialMovingAverage moving_average(1.0 / (10));

        int64_t train_epoc_index = 0;
        int64_t number_of_training_epocs = 100;
        double best_test_loss = std::numeric_limits<double>::max();
        double final_test_loss = std::numeric_limits<double>::max();
        std::array<double, 2> xDims{std::numeric_limits<double>::max(),std::numeric_limits<double>::min()};
        std::array<double, 2> yDims{std::numeric_limits<double>::max(),std::numeric_limits<double>::min()};
        int64_t iteration = 0;
        for(; train_epoc_index < number_of_training_epocs; iteration++)
        {
        //Load data into blobs
        bool last_example_in_train_epoc = training_data_source.ReadBlob((char *) input_blob.mutable_data<int8_t>(),
                                                                  (char *) expected_output_blob.mutable_data<float>());
        //Run network with loaded instance
        shape_2d_localize_train_net->Run();

        //Get loss exponentially weighted moving average
        caffe2::TensorCPU& loss = GoodBot::GetMutableTensor("shape_2d_localize_avg_loss_cpu", workspace);

        caffe2::TensorCPU& shape_2d_localize_fc_output_cpu = GoodBot::GetMutableTensor("shape_2d_localize_fc_output_cpu", workspace);

        moving_average.Update((double) *loss.mutable_data<float>());

        xDims[0] = std::min<double>(xDims[0], expected_output_blob.mutable_data<float>()[0]);
        xDims[1] = std::max<double>(xDims[1], expected_output_blob.mutable_data<float>()[0]);

        yDims[0] = std::min<double>(yDims[0], expected_output_blob.mutable_data<float>()[1]);
        yDims[1] = std::max<double>(yDims[1], expected_output_blob.mutable_data<float>()[1]);

        //std::cout << "Moving average loss ( " << iteration << " ): " << moving_average.GetAverage() << " " << expected_output_blob.mutable_data<float>()[0] << " " << expected_output_blob.mutable_data<float>()[1]
        //          << " " << shape_2d_localize_fc_output_cpu.mutable_data<float>()[0] << " " << shape_2d_localize_fc_output_cpu.mutable_data<float>()[1] << " " << train_epoc_index << std::endl;

        if(last_example_in_train_epoc)
        {
            //Get the average test error
            bool last_example_in_test_epoc = false;
            int64_t test_epoc_example_count = 0;
            double average_test_loss = 0.0;
            while(!last_example_in_test_epoc)
            {
                last_example_in_test_epoc = test_data_source.ReadBlob((char *) input_blob.mutable_data<int8_t>(),
                                                                              (char *) expected_output_blob.mutable_data<float>());
                //Run network with loaded instance
                shape_2d_localize_test_net->Run();

                average_test_loss += *loss.mutable_data<float>();

                test_epoc_example_count++;
            }
            average_test_loss = average_test_loss / test_epoc_example_count;

            final_test_loss = average_test_loss;
            best_test_loss = std::min(average_test_loss, best_test_loss);
            std::cout << "Average test loss: " << average_test_loss << std::endl;

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

        logger.AddEntry(entry);

        return best_test_loss;
    };


    //TestHyperParameter({.00019542143346}, {3, 1092, 1, 37, 3});

    //Pass in parameters:
    //doubles:
    //Learning rate: [.001, .00001]

    //integers:
    //#Fully connected rectified layers (after conv layers): [0,6]
    //# nodes in fully connected layers: [10, 2000]
    //#Convolution models: [0, 6]
    //# filters at base layer: [1, 200]
    //Stride level: [1,4]

    GoodBot::InvestigatorRunConstraints constraints;
    constraints.IntegerConstraints.emplace_back(0,6);
    constraints.IntegerConstraints.emplace_back(10, 2000);
    constraints.IntegerConstraints.emplace_back(0,6);
    constraints.IntegerConstraints.emplace_back(1,50);
    constraints.IntegerConstraints.emplace_back(1,4);

    constraints.DoubleConstraints.emplace_back(0.00001, .001);

    constraints.MaxRunTimeMilliSeconds = 1000*60*60*24*7; //Go for about a week

    GoodBot::Investigator experimenter(constraints, TestHyperParameter);

    experimenter.Optimize();

return 0;
}
