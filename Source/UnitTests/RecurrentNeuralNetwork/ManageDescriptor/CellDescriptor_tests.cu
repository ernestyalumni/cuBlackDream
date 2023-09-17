#include "RecurrentNeuralNetwork/ManageDescriptor/CellDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::ManageDescriptor::CellDescriptor;
using RecurrentNeuralNetwork::DefaultParameters;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CellDescriptorTests, ConstructsWithParametersForLSTMUnidirectional)
{
  DefaultParameters parameters {};
  parameters.cell_mode_ = CUDNN_LSTM;
  parameters.input_size_ = 784;
  parameters.hidden_size_ = 100;
  parameters.projection_size_ = 10;
  parameters.maximum_sequence_length_ = 28;
  parameters.batch_size_ = 100;


  CellDescriptor<3> cell_descriptor {parameters};

  EXPECT_EQ(
    cell_descriptor.set_for_ND_tensor_.get_dimensions_array_value(0),
    2);

  EXPECT_EQ(
    cell_descriptor.set_for_ND_tensor_.get_dimensions_array_value(1),
    100);

  EXPECT_EQ(
    cell_descriptor.set_for_ND_tensor_.get_dimensions_array_value(2),
    100);

  EXPECT_EQ(
    cell_descriptor.set_for_ND_tensor_.get_strides_array_value(0),
    10000);

  EXPECT_EQ(
    cell_descriptor.set_for_ND_tensor_.get_strides_array_value(1),
    100);

  EXPECT_EQ(
    cell_descriptor.set_for_ND_tensor_.get_strides_array_value(2),
    1);
}


} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests