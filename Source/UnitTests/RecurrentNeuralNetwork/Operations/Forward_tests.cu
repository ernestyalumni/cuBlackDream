#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::SequenceLengthArray;
using RecurrentNeuralNetwork::HostSequenceLengthArray;
using RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::WeightSpace;
using RecurrentNeuralNetwork::WorkAndReserveSpaces;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace Operations
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ForwardTests, Constructs)
{
  DefaultParameters parameters {};
  SequenceLengthArray sequence_length_array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  sequence_length_array.copy_host_input_to_device(host_array);

  InputDescriptor x_descriptor {parameters, sequence_length_array};
  LibraryHandleDropoutRNN descriptors {parameters};
  WeightSpace weight_space {descriptors};
  WorkAndReserveSpaces spaces {descriptors, x_descriptor};
}

} // namespace Operations
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests