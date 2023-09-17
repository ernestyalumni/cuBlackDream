#include "RecurrentNeuralNetwork/ManageDescriptor/HiddenDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor3Dim;
using RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor;
using RecurrentNeuralNetwork::DefaultParameters;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptorTests, DefaultConstructs)
{
  HiddenDescriptor<4> hidden_descriptor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptorTests, Destructs)
{
  {
    HiddenDescriptor<4> hidden_descriptor {};

    SUCCEED();
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptorTests, ConstructsWithParameters)
{
  DefaultParameters parameters {};
  HiddenDescriptor<4> hidden_descriptor {parameters};

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(0),
    2);
  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(1),
    64);
    EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(2),
    512);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptorTests, SetStridesByDimensionsSets)
{
  DefaultParameters parameters {};
  HiddenDescriptor<3> hidden_descriptor {parameters};

  hidden_descriptor.set_strides_by_dimensions();

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(0),
    32768);
  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(1),
    512);
    EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(2),
    1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptorTests, SetDescriptorSetsAfterSettingStrides)
{
  DefaultParameters parameters {};
  HiddenDescriptor<3> hidden_descriptor {parameters};
  hidden_descriptor.set_strides_by_dimensions();

  const auto result = hidden_descriptor.set_descriptor(parameters);

  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptorTests, SetDescriptorSets)
{
  DefaultParameters parameters {};

  HiddenDescriptor<3> hidden_descriptor {};
  hidden_descriptor.set_for_ND_tensor_.set_dimensions_array_value(
    0,
    parameters.number_of_layers_ * parameters.get_bidirectional_scale());
  hidden_descriptor.set_for_ND_tensor_.set_dimensions_array_value(
    1,
    parameters.batch_size_);
  hidden_descriptor.set_for_ND_tensor_.set_dimensions_array_value(
    2,
    parameters.hidden_size_);

  hidden_descriptor.set_for_ND_tensor_.set_strides_array_value(
    0,
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(1) *
      hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(2));
  hidden_descriptor.set_for_ND_tensor_.set_strides_array_value(
    1,
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(2));
  hidden_descriptor.set_for_ND_tensor_.set_strides_array_value(2, 1);

  const auto result = hidden_descriptor.set_descriptor(parameters);
  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptor3DimTests, ConstructsWithParameters)
{
  DefaultParameters parameters {};
  HiddenDescriptor3Dim hidden_descriptor {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  HiddenDescriptor3DimTests,
  ConstructsWithParametersSetForTanhUnidirectional)
{
  DefaultParameters parameters {};
  parameters.cell_mode_ = CUDNN_RNN_TANH;
  parameters.input_size_ = 784;
  parameters.hidden_size_ = 100;
  parameters.projection_size_ = 10;
  parameters.maximum_sequence_length_ = 28;
  parameters.batch_size_ = 100;


  HiddenDescriptor<3> hidden_descriptor {parameters};

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(0),
    2);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(1),
    100);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(2),
    100);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(0),
    10000);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(1),
    100);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(2),
    1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptor3DimTests, ConstructsWithParametersSetForGRUBidirectional)
{
  DefaultParameters parameters {};
  parameters.cell_mode_ = CUDNN_GRU;
  parameters.direction_mode_ = CUDNN_BIDIRECTIONAL;
  parameters.input_size_ = 784;
  parameters.hidden_size_ = 100;
  parameters.projection_size_ = 10;
  parameters.maximum_sequence_length_ = 28;
  parameters.batch_size_ = 100;

  HiddenDescriptor<3> hidden_descriptor {parameters};

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(0),
    4);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(1),
    100);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(2),
    100);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(0),
    10000);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(1),
    100);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(2),
    1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HiddenDescriptor3DimTests, ConstructsWithParametersForLSTMUnidirectional)
{
  DefaultParameters parameters {};
  parameters.cell_mode_ = CUDNN_LSTM;
  parameters.input_size_ = 784;
  parameters.hidden_size_ = 100;
  parameters.projection_size_ = 10;
  parameters.maximum_sequence_length_ = 28;
  parameters.batch_size_ = 100;


  HiddenDescriptor<3> hidden_descriptor {parameters};

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(0),
    2);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(1),
    100);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_dimensions_array_value(2),
    10);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(0),
    1000);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(1),
    10);

  EXPECT_EQ(
    hidden_descriptor.set_for_ND_tensor_.get_strides_array_value(2),
    1);
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests