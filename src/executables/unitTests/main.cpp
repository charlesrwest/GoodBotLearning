#define CATCH_CONFIG_MAIN //Make main function automatically
#include "catch.hpp"
#include<cstdlib>
#include<string>

#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include<google/protobuf/text_format.h>
#include<iostream>
#include<cmath>
#include<cassert>
#include<fstream>
#include<random>
#include<cstdio>
#include "RandomizedFileDataLoader.hpp"
#include "SOMScopeGuard.hpp"
#include "SOMException.hpp"
#include<nlopt.hpp>

#include "ExponentialMovingAverage.hpp"
#include "NetConstruction.hpp"
#include "TestHelpers.hpp"
#include "MemoryDataLoader.hpp"
#include "ExperimentLogger.hpp"
#include "SQLite3Wrapper.hpp"
#include "SOMScopeGuard.hpp"
#include "Optimizer.hpp"
#include "UtilityFunctions.hpp"

TEST_CASE("Test training data file creation, splitting and reading")
{
    char base_character = 'A';

    int64_t input_blob_size = 10;
    int64_t output_blob_size = 5;
    int64_t batch_size = 2;

    //Make training data file
    {
    std::ofstream output_file("AlphabetTrainingData.blobber", std::ofstream::binary);
    for(int64_t letter_index = 0; letter_index < (26/2); letter_index++)
    {
        std::vector<char> input_blob(input_blob_size, base_character+letter_index*2);
        std::vector<char> output_blob(output_blob_size, base_character+letter_index*2+1);

        output_file.write(input_blob.data(), input_blob.size());
        output_file.write(output_blob.data(), output_blob.size());
    }
    }

    //Split the training data into 2 files (train, test)
    GoodBot::SplitBlobberFile(.8,  input_blob_size + output_blob_size, batch_size,
                          "AlphabetTrainingData.blobber", "AlphabetTrainingDataTrain.blobber", "AlphabetTrainingDataTest.blobber");

}


TEST_CASE("Test non-linear optimization with 1D doubles")
{
    std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> objective_function = [](const std::vector<double>& doubleParameters, const std::vector<int64_t>&)
    {
        return pow((doubleParameters[0]-1), 2.0); //Parabola with zero at 1
    };

    for(int64_t optimizer_test = 0; optimizer_test < 1000; optimizer_test++)
    {
        GoodBot::Optimizer opt(objective_function, {}, {{-10.0, 10.0}}, {}, 20, .5);

        for(int64_t step = 0; step < 100000; step++)
        {
            opt.StepSearch();
            std::tuple<double, std::vector<int64_t>, std::vector<double>> result = opt.GetBestParameters();
            if(std::get<0>(result) == 0)
            {
                //Got best possible value, so stop searching
                break;
            }
            //std::cout << "Min " << std::get<0>(result) << std::endl;
        }

        std::tuple<double, std::vector<int64_t>, std::vector<double>> result = opt.GetBestParameters();
        //std::cout << "Min " << std::get<0>(result) << std::endl;
        REQUIRE(std::get<0>(result) >= 0.0);
        REQUIRE(std::get<0>(result) < .1);
    }

}

TEST_CASE("Test non-linear optimization with 1D integer")
{
    std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> objective_function = [](const std::vector<double>& doubleParameters, const std::vector<int64_t>& integerParameters)
    {
        return pow((integerParameters[0]-1), 2.0); //Parabola with zero at 1
    };

    for(int64_t optimizer_test = 0; optimizer_test < 1000; optimizer_test++)
    {
    GoodBot::Optimizer opt(objective_function, {{-1000, 1000}}, {}, {}, 20, .5);

    for(int64_t step = 0; step < 100000; step++)
    {
        opt.StepSearch();
        std::tuple<double, std::vector<int64_t>, std::vector<double>> result = opt.GetBestParameters();
        //std::cout << "Min " << std::get<0>(result) << std::endl;std::cout << "Min " << std::get<0>(result) << std::endl;
        if(std::get<0>(result) == 0)
        {
            //Got best possible value, so stop searching
            break;
        }
    }

    std::tuple<double, std::vector<int64_t>, std::vector<double>> result = opt.GetBestParameters();
    //std::cout << "Min " << std::get<0>(result) << std::endl;
    REQUIRE(std::get<0>(result) >= 0.0);
    REQUIRE(std::get<0>(result) == 0.0);
    }
}

TEST_CASE("Test non-linear optimization with 2D integer/double")
{
    std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> objective_function = [](const std::vector<double>& doubleParameters, const std::vector<int64_t>& integerParameters)
    {
        return pow((integerParameters[0]-1), 2.0) + pow((doubleParameters[0]-1), 2.0); //Parabola with zero at 1
    };

    for(int64_t optimizer_test = 0; optimizer_test < 1000; optimizer_test++)
    {
        GoodBot::Optimizer opt(objective_function, {{-1000, 1000}}, {{-10.0, 10.0}}, {}, 40, .3);

    for(int64_t step = 0; step < 100000; step++)
    {
        opt.StepSearch();
        std::tuple<double, std::vector<int64_t>, std::vector<double>> result = opt.GetBestParameters();
        //std::cout << "Min " << std::get<0>(result) << std::endl;std::cout << "Min " << std::get<0>(result) << std::endl;
        if(std::get<0>(result) == 0)
        {
            //Got best possible value, so stop searching
            break;
        }
    }

    std::tuple<double, std::vector<int64_t>, std::vector<double>> result = opt.GetBestParameters();
    std::cout << "Min " << std::get<0>(result) << std::endl;
    REQUIRE(std::get<0>(result) >= 0.0);
    REQUIRE(std::get<0>(result) < 0.1);
    }
}

TEST_CASE("Test experiment logger", "Logger")
{
    SOMScopeGuard file_guard([](){remove("testLoggerTmp.db");});

    std::unique_ptr<GoodBot::ExperimentLogger> logger;

    SOM_TRY
            logger.reset(new GoodBot::ExperimentLogger("testLoggerTmp.db"));
    SOM_CATCH("Error initializing logger")

    std::vector<GoodBot::LogEntry> test_entries;
    test_entries.emplace_back();
    test_entries[0].ExperimentGroupName = "ExperimentGroupName1";
    test_entries[0].DataSetName = "DataSetName1";
    test_entries[0].InvestigationMethod = "InvestigationMethod1";
    test_entries[0].BestTestError = 1.0;
    test_entries[0].FinalTestError = 2.0;
    test_entries[0].TimeOfCompletion = 3;
    test_entries[0].RunTime = 4;
    test_entries[0].NumberOfTrainingIterations = 5;
    test_entries[0].NumberOfTrainingEpocs = 6;
    test_entries[0].HashOfNetspace = "Hash1";
    test_entries[0].NetspaceSummary = "NetSpaceSummary1";

    test_entries[0].IntegerHyperParameters["IntBob"] = {7,8,9};
    test_entries[0].IntegerHyperParameters["IntSaga"] = {10,11,12};
    test_entries[0].DoubleHyperParameters["DoubleBob"] = {13.0,14.0,15.0};
    test_entries[0].StringHyperParameters["StringBob"] = {"test","mellon", "sprite"};

    SOM_TRY
         logger->AddEntry(test_entries[0]);
    SOM_CATCH("Error logging entry")

    {
        CRWUtility::SQLITE3::DatabaseConnection Session("testLoggerTmp.db");
        std::vector<GoodBot::LogEntry> retrieved_entries = GoodBot::RetrieveAllEntries(Session);

        REQUIRE(retrieved_entries.size() == test_entries.size());
        for(int64_t entry_index = 0; entry_index < retrieved_entries.size(); entry_index++)
        {
            REQUIRE(retrieved_entries[entry_index] == test_entries[entry_index]);
        }
    }

    test_entries.emplace_back();
    test_entries[1].ExperimentGroupName = "ExperimentGroupName2";
    test_entries[1].DataSetName = "DataSetName2";
    test_entries[1].InvestigationMethod = "InvestigationMethod2";
    test_entries[1].BestTestError = 16.0;
    test_entries[1].FinalTestError = 17.0;
    test_entries[1].TimeOfCompletion = 18;
    test_entries[1].RunTime = 19;
    test_entries[1].NumberOfTrainingIterations = 20;
    test_entries[1].NumberOfTrainingEpocs = 21;
    test_entries[1].HashOfNetspace = "Hash22";
    test_entries[1].NetspaceSummary = "NetSpaceSummary23";

    test_entries[1].IntegerHyperParameters["Int"] = {24,25};
    test_entries[1].DoubleHyperParameters["DoubleBob"] = {26.0,27.0};
    test_entries[1].DoubleHyperParameters["DoubleBob"] = {28.0,29.0};
    test_entries[1].StringHyperParameters["StringBob"] = {"liz","Bello", "Tut"};

    SOM_TRY
         logger->AddEntry(test_entries[1]);
    SOM_CATCH("Error logging entry")

    {
        CRWUtility::SQLITE3::DatabaseConnection Session("testLoggerTmp.db");
        std::vector<GoodBot::LogEntry> retrieved_entries = GoodBot::RetrieveAllEntries(Session);

        REQUIRE(retrieved_entries.size() == test_entries.size());
        for(int64_t entry_index = 0; entry_index < retrieved_entries.size(); entry_index++)
        {
            REQUIRE(retrieved_entries[entry_index] == test_entries[entry_index]);
        }
    }
}

TEST_CASE("Draw shapes", "[Example]")
{
    std::vector<int32_t> labels;
    std::vector<PseudoImage<char>> images;

    std::tie(labels, images) = CreateShapeCategorizationImageTrainingData<char>(0, 100, 1, {0});

    REQUIRE(labels.size() > 0);
    REQUIRE(labels.size() == images.size());

    //VisualizeTrainingData<char>(labels, images, 0);
}

TEST_CASE("Simple localization conv network", "[Example]")
{
    //Make function capatable with investigator framework
    std::function<double(const std::vector<double>&, const std::vector<int64_t>&)> TestHyperParameter =
            [](const std::vector<double>& doubleParameters, const std::vector<int64_t>& integerParameters)
    {
        //We expect one integer parameter which indicates the image depth to use
        SOM_ASSERT(integerParameters.size() == 1, "Expected single integer parameter");
        SOM_ASSERT(doubleParameters.size() == 0, "Expected no double parameters");

        int64_t input_depth = integerParameters[0];

        //Create the Caffe2 workspace/context
        caffe2::Workspace workspace;
        caffe2::CPUContext context;

        GoodBot::NetSpace netspace(workspace);

        /** Create inputs/outputs */

        //Batch size, channel depth, width/height
        GoodBot::AddConstantFillOp("shape_2d_localize_input", "input_blob", 0,  caffe2::TensorProto::INT8, {1, input_depth, 20, 20}, {"INIT"}, false, netspace);

        //Batch size, expected category
        GoodBot::AddConstantFillOp("shape_2d_localize_expected_output", "expected_output_blob", 0.0f,  caffe2::TensorProto::FLOAT, {1, 2}, {"INIT"}, false, netspace);

        //Produce the training data
        std::vector<std::array<float, 2>> training_expected_output;
        std::vector<PseudoImage<char>> images;

        //If we have a depth of more than one, make a copy of the training data with exactly one of the depth layers drawn on for each possible depth.
        for(int64_t fill_depth_index = 0; fill_depth_index < input_depth; fill_depth_index++)
        {
            std::vector<std::array<float, 2>> training_expected_output_buffer;
            std::vector<PseudoImage<char>> images_buffer;

            std::tie(training_expected_output_buffer, images_buffer) = CreateShape2DLocalizationImageTrainingData<char>(0, 100, input_depth, 20, {fill_depth_index});

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

        //Make the training network
        GoodBot::AddCastOp("shape_2d_localize_cast", "input_blob", "INT8", "input_blob_casted", "FLOAT", {}, netspace);
        AddScaleOp("shape_2d_localize_scale", "input_blob_casted", "input_blob_scaled", (1.0/128.0), {}, netspace);

        //conv (3x3, 20 channels)
        //conv (3x3, 20 channels)
        //Relu
        AddConvModule("shape_2d_localize_conv_1", "input_blob_scaled", "shape_2d_localize_conv_1", 32, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
        AddConvModule("shape_2d_localize_conv_2", "shape_2d_localize_conv_1", "shape_2d_localize_conv_2", 32, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
        AddReluOp("shape_2d_localize_conv_2_relu_1", "shape_2d_localize_conv_2", "shape_2d_localize_conv_2", {}, netspace);

        //conv (3x3, 20 channels)
        //conv (3x3, 20 channels)
        //Relu
        AddConvModule("shape_2d_localize_conv_3", "shape_2d_localize_conv_2", "shape_2d_localize_conv_3", 64, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
        AddConvModule("shape_2d_localize_conv_4", "shape_2d_localize_conv_3", "shape_2d_localize_conv_4", 64, 1, 1, 3, "XavierFill", "ConstantFill", netspace);
        AddReluOp("shape_2d_localize_conv_4_relu_2", "shape_2d_localize_conv_4", "shape_2d_localize_conv_4", {}, netspace);

        //relu fc 500
        //relu fc 500
        //fc 2
        //softmax
        AddFullyConnectedModuleWithActivation("shape_2d_localize_fc_1", "shape_2d_localize_conv_4", "shape_2d_localize_fc_1", 512, "Relu", "XavierFill", "ConstantFill", netspace);
        AddFullyConnectedModuleWithActivation("shape_2d_localize_fc_2", "shape_2d_localize_fc_1", "shape_2d_localize_fc_2", 512, "Relu", "XavierFill", "ConstantFill", netspace);
        AddFullyConnectedModule("shape_2d_localize_fc_3", "shape_2d_localize_fc_2", "shape_2d_localize_fc_3", 2, "XavierFill", "ConstantFill", netspace);

        //Make loss/loopback for gradient
        AddSquaredL2DistanceOp("shape_2d_localize_loss_l2_dist", "shape_2d_localize_fc_3", "expected_output_blob", "shape_2d_localize_loss_l2_dist", {"TRAIN", "TEST"}, netspace);
        AddAveragedLossOp("shape_2d_localize_avg_loss", "shape_2d_localize_loss_l2_dist", "shape_2d_localize_avg_loss", {"TRAIN", "TEST"}, netspace);
        AddNetworkGradientLoopBack("shape_2d_localize_gradient_loop_back", "shape_2d_localize_avg_loss", {"TRAIN", "TEST"}, netspace);

        //Add gradient ops
        AddGradientOperators("shape_2d_localize", {"TRAIN", "TEST"}, netspace);

        //Add solver ops -> lower than .001 learning rate n
        GoodBot::AddAdamSolvers("shape_2d_localize", netspace, .9, .999, 1e-5, -.0001);

        //Initialize network
        caffe2::NetDef shape_2d_localize_init_def = GoodBot::GetNetwork("shape_2d_localize", "INIT", false, netspace);
        caffe2::NetBase* shape_2d_localize_init_net = workspace.CreateNet(shape_2d_localize_init_def);
        shape_2d_localize_init_net->Run();

        REQUIRE(BlobNamesFound({"input_blob", "expected_output_blob"}, workspace));
        REQUIRE(BlobShapeMatches("input_blob", {1, input_depth, 20, 20}, workspace));
        REQUIRE(BlobShapeMatches("expected_output_blob", {1, 2}, workspace));

        caffe2::TensorCPU& input_blob = GoodBot::GetMutableTensor("input_blob", workspace);
        caffe2::TensorCPU& expected_output_blob = GoodBot::GetMutableTensor("expected_output_blob", workspace);


        //Create training network
        caffe2::NetDef shape_2d_localize_train_def = GoodBot::GetNetwork("shape_2d_localize", "TRAIN", true, netspace);
        caffe2::NetBase* shape_2d_localize_train_net = workspace.CreateNet(shape_2d_localize_train_def);

        //Create test network
        caffe2::NetDef shape_2d_localize_test_def = GoodBot::GetNetwork("shape_2d_localize", "TEST", true, netspace);
        caffe2::NetBase* shape_2d_localize_test_net = workspace.CreateNet(shape_2d_localize_test_def);

        GoodBot::ExponentialMovingAverage moving_average(1.0 / (10));

        int64_t number_of_training_iterations = 30000;
        int64_t train_epoc_index = 0;
        int64_t number_of_training_epocs = 10;
        double best_test_loss = std::numeric_limits<double>::max();
        for(int64_t iteration = 0; train_epoc_index < number_of_training_epocs; iteration++)
        {
        //Load data into blobs
        bool last_example_in_train_epoc = training_data_source.ReadBlob((char *) input_blob.mutable_data<int8_t>(),
                                                                  (char *) expected_output_blob.mutable_data<float>());
        //Run network with loaded instance
        shape_2d_localize_train_net->Run();

        //Get loss exponentially weighted moving average
        caffe2::TensorCPU& loss = GoodBot::GetMutableTensor("shape_2d_localize_avg_loss", workspace);

        caffe2::TensorCPU& shape_2d_localize_fc_3 = GoodBot::GetMutableTensor("shape_2d_localize_fc_3", workspace);

        moving_average.Update((double) *loss.mutable_data<float>());

        std::cout << "Moving average loss ( " << iteration << " ): " << moving_average.GetAverage() << " " << expected_output_blob.mutable_data<float>()[0] << " " << expected_output_blob.mutable_data<float>()[1]
                  << " " << shape_2d_localize_fc_3.mutable_data<float>()[0] << " " << shape_2d_localize_fc_3.mutable_data<float>()[1] << " " << train_epoc_index << std::endl;

        if(last_example_in_train_epoc)
        {
            //Get the average test error
            bool last_example_in_test_epoc = false;
            int64_t test_epoc_count = 0;
            double average_test_loss = 0.0;
            while(!last_example_in_test_epoc)
            {
                last_example_in_test_epoc = test_data_source.ReadBlob((char *) input_blob.mutable_data<int8_t>(),
                                                                              (char *) expected_output_blob.mutable_data<float>());
                //Run network with loaded instance
                shape_2d_localize_test_net->Run();

                average_test_loss += *loss.mutable_data<float>();

                test_epoc_count++;
            }
            average_test_loss = average_test_loss / test_epoc_count;

            best_test_loss = std::min(average_test_loss, best_test_loss);
            std::cout << "Average test loss: " << average_test_loss << std::endl;

            train_epoc_index++;
        }
        }
        REQUIRE(best_test_loss < .01);
        REQUIRE(moving_average.GetAverage() < .01);

        return best_test_loss;
    };

    //Loop through different input depths
    for(int64_t input_depth : {1, 2, 3})
    {
        TestHyperParameter({}, {input_depth});
    }
}


TEST_CASE("Simple categorization conv network", "[Example]")
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

        std::tie(labels_buffer, images_buffer) = CreateShapeCategorizationImageTrainingData<char>(0, 100, input_depth, {fill_depth_index});

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
GoodBot::RandomizedFileDataLoader loader(temp_file_name, input_blob_size, output_blob_size, bufferSize, numberOfBuffers);

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


