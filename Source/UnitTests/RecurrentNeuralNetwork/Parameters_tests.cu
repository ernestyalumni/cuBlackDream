#include "RecurrentNeuralNetwork/Parameters.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::DefaultParameters;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DefaultParametersTests, Constructs)
{
  DefaultParameters parameters {};

  // CUDNN_RNN_ALGO_STANDARD
  EXPECT_EQ(static_cast<int>(parameters.algo_), 0);
  // CUDNN_RNN_RELU
  EXPECT_EQ(static_cast<int>(parameters.cell_mode_), 0);
  // CUDNN_RNN_DOUBLE_BIAS
  EXPECT_EQ(static_cast<int>(parameters.bias_mode_), 2);
  EXPECT_EQ(parameters.bias_mode_, CUDNN_RNN_DOUBLE_BIAS);
  // CUDNN_UNIDIRECTIONAL
  EXPECT_EQ(static_cast<int>(parameters.direction_mode_), 0);
  // CUDNN_DATA_FLOAT
  EXPECT_EQ(static_cast<int>(parameters.data_type_), 0);
  EXPECT_EQ(parameters.data_type_, CUDNN_DATA_FLOAT);

  EXPECT_EQ(parameters.batch_size_, 64);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DefaultParametersTests, GetBidirectionalScaleGetsScale)
{
  {
    DefaultParameters parameters {};

    EXPECT_EQ(parameters.get_bidirectional_scale(), 1);
  }
  {
    DefaultParameters parameters {};

    parameters.direction_mode_ = CUDNN_BIDIRECTIONAL;

    EXPECT_EQ(parameters.get_bidirectional_scale(), 2);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DefaultParametersTests, GetInputTensorSizeGetsSize)
{
  {
    DefaultParameters parameters {};

    EXPECT_EQ(parameters.get_input_tensor_size(), 655360);
  }
  {
    DefaultParameters parameters {};
    parameters.cell_mode_ = CUDNN_GRU;
    // 512 cells * 6 parameters per cell.
    parameters.input_size_ = 512 * 6;
    parameters.hidden_size_ = 100;
    parameters.projection_size_ = 100;
    parameters.maximum_sequence_length_ = 4;
    parameters.batch_size_ = 10;

    EXPECT_EQ(parameters.get_input_tensor_size(), 512 * 6 * 4 * 10);    
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DefaultParametersTests, GetOutputTensorSizeGetsSize)
{
  {
    DefaultParameters parameters {};

    EXPECT_EQ(parameters.get_output_tensor_size(), 655360);
  }
  {
    DefaultParameters parameters {};
    parameters.cell_mode_ = CUDNN_GRU;
    // 512 cells * 6 parameters per cell.
    parameters.input_size_ = 512 * 6;
    parameters.hidden_size_ = 100;
    parameters.projection_size_ = 100;
    parameters.maximum_sequence_length_ = 4;
    parameters.batch_size_ = 10;

    EXPECT_EQ(parameters.get_output_tensor_size(), 100 * 4 * 10);    
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DefaultParametersTests, GetHiddenTensorSizeGetsSize)
{
  {
    DefaultParameters parameters {};

    EXPECT_EQ(parameters.get_hidden_tensor_size(), 65536);
  }
  {
    DefaultParameters parameters {};
    parameters.cell_mode_ = CUDNN_GRU;
    // 512 cells * 6 parameters per cell.
    parameters.input_size_ = 512 * 6;
    parameters.hidden_size_ = 100;
    parameters.projection_size_ = 100;
    parameters.maximum_sequence_length_ = 4;
    parameters.batch_size_ = 10;

    EXPECT_EQ(parameters.get_hidden_tensor_size(), 2 * 10 * 100);    
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DefaultParametersTests, GetTotalMemoryConsumptionGetsSize)
{
  DefaultParameters parameters {};

  EXPECT_EQ(parameters.get_total_memory_consumption<float>(), 12582912);
}

} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests