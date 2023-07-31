#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::HostSequenceLengthArray;
using RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::SequenceLengthArray;
using RecurrentNeuralNetwork::WorkAndReserveSpaces;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WorkAndReserveSpaces, ConstructsWithLibraryHandleDropoutRNN)
{
  DefaultParameters parameters {};
  SequenceLengthArray sequence_length_array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  sequence_length_array.copy_host_input_to_device(host_array);

  InputDescriptor x_descriptor {parameters, sequence_length_array};
  LibraryHandleDropoutRNN descriptors {parameters};

  WorkAndReserveSpaces spaces {descriptors, x_descriptor};

  EXPECT_EQ(spaces.get_forward_mode(), CUDNN_FWD_MODE_TRAINING);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WorkAndReserveSpaces, GetWorkSpaceSizeGetsSize)
{
  DefaultParameters parameters {};
  SequenceLengthArray sequence_length_array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  sequence_length_array.copy_host_input_to_device(host_array);

  InputDescriptor x_descriptor {parameters, sequence_length_array};
  LibraryHandleDropoutRNN descriptors {parameters};

  WorkAndReserveSpaces spaces {descriptors, x_descriptor};

  EXPECT_EQ(spaces.get_work_space_size(), 9175040);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WorkAndReserveSpaces, GetReserveSpaceSizeGetsSize)
{
  DefaultParameters parameters {};
  SequenceLengthArray sequence_length_array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  sequence_length_array.copy_host_input_to_device(host_array);

  InputDescriptor x_descriptor {parameters, sequence_length_array};
  LibraryHandleDropoutRNN descriptors {parameters};

  WorkAndReserveSpaces spaces {descriptors, x_descriptor};

  EXPECT_EQ(spaces.get_reserve_space_size(), 7864336);
}

} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests