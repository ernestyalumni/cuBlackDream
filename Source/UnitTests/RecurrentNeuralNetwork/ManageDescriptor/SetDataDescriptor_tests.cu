#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/SetDataDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "gtest/gtest.h"

using RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::SetDataDescriptor;
using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::HostSequenceLengthArray;
using RecurrentNeuralNetwork::SequenceLengthArray;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDataDescriptor, DefaultConstructs)
{
  SetDataDescriptor set_data_descriptor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDataDescriptor, SetDescriptorForInputSets)
{
  DataDescriptor x_data_descriptor {};
  SetDataDescriptor set_data_descriptor {};

  DefaultParameters parameters {};
  SequenceLengthArray array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  array.copy_host_input_to_device(host_array);

  const auto result = set_data_descriptor.set_descriptor_for_input(
    x_data_descriptor,
    parameters,
    array);

  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDataDescriptor, SetDescriptorForOutputSets)
{
  DataDescriptor y_data_descriptor {};
  SetDataDescriptor set_data_descriptor {};

  DefaultParameters parameters {};
  SequenceLengthArray array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  array.copy_host_input_to_device(host_array);

  const auto result = set_data_descriptor.set_descriptor_for_output(
    y_data_descriptor,
    parameters,
    array);

  EXPECT_TRUE(result.is_success());
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests