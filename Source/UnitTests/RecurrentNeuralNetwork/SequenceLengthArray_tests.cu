#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <vector>

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::HostSequenceLengthArray;
using RecurrentNeuralNetwork::SequenceLengthArray;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostSequenceLengthArrayTests, ConstructsWithParameters)
{
  DefaultParameters parameters {};

  HostSequenceLengthArray host_array {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostSequenceLengthArrayTests, CopyValuesCopiesFromStdVector)
{
  DefaultParameters parameters {};

  HostSequenceLengthArray host_array {parameters};

  std::vector<int> values (
    parameters.batch_size_,
    parameters.maximum_sequence_length_);

  host_array.copy_values(values);

  for (std::size_t i {0}; i < parameters.batch_size_; ++i)
  {
    EXPECT_EQ(host_array.sequence_length_array_[i], 20);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostSequenceLengthArrayTests, SetAllToMaximumSequenceLengthSets)
{
  DefaultParameters parameters {};

  HostSequenceLengthArray host_array {parameters};

  host_array.set_all_to_maximum_sequence_length();

  for (std::size_t i {0}; i < parameters.batch_size_; ++i)
  {
    EXPECT_EQ(host_array.sequence_length_array_[i], 20);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SequenceLengthArrayTests, ConstructsWithParameters)
{
  DefaultParameters parameters {};

  SequenceLengthArray array {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SequenceLengthArrayTests, CopyHostInputToDeviceCopies)
{
  DefaultParameters parameters {};

  SequenceLengthArray array {parameters};

  HostSequenceLengthArray host_array {parameters};

  host_array.set_all_to_maximum_sequence_length();

  array.copy_host_input_to_device(host_array);

  SUCCEED();
}

} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests