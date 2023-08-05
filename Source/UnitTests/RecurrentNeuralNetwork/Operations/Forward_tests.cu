#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/OutputDescriptor.h"
#include "RecurrentNeuralNetwork/Modules/Hidden.h"
#include "RecurrentNeuralNetwork/Modules/Input.h"
#include "RecurrentNeuralNetwork/Modules/Output.h"
#include "RecurrentNeuralNetwork/Operations/forward.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "gtest/gtest.h"

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::HostSequenceLengthArray;
using RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor3Dim;
using RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor;
using RecurrentNeuralNetwork::Modules::Hidden;
using RecurrentNeuralNetwork::Modules::Input;
using RecurrentNeuralNetwork::Modules::Output;
using RecurrentNeuralNetwork::Operations::forward;
using RecurrentNeuralNetwork::SequenceLengthArray;
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
TEST(ForwardTests, Forwards)
{
  DefaultParameters parameters {};
  SequenceLengthArray sequence_length_array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  sequence_length_array.copy_host_input_to_device(host_array);

  InputDescriptor x_descriptor {parameters, sequence_length_array};
  OutputDescriptor y_descriptor {parameters, sequence_length_array};
  LibraryHandleDropoutRNN descriptors {parameters};
  HiddenDescriptor<3> h_descriptor {parameters};
  HiddenDescriptor<3> c_descriptor {parameters};
  h_descriptor.set_strides_by_dimensions();
  c_descriptor.set_strides_by_dimensions();
  h_descriptor.set_descriptor(parameters); 
  c_descriptor.set_descriptor(parameters); 

  WeightSpace weight_space {descriptors};
  WorkAndReserveSpaces spaces {descriptors, x_descriptor};

  Input<float> x {parameters};
  Output<float> y {parameters};
  Hidden<float> hx {parameters};
  Hidden<float> hy {parameters};
  Hidden<float> cx {parameters};
  Hidden<float> cy {parameters};

  const auto result = forward<float, 3>(
    descriptors,
    sequence_length_array,
    x_descriptor,
    x,
    y_descriptor,
    y,
    h_descriptor,
    hx,
    hy,
    c_descriptor,
    cx,
    cy,
    weight_space,
    spaces);

  EXPECT_TRUE(result.is_success());
}

} // namespace Operations
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests