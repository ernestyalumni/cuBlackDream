#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/OutputDescriptor.h"
#include "RecurrentNeuralNetwork/Modules/Hidden.h"
#include "RecurrentNeuralNetwork/Modules/Input.h"
#include "RecurrentNeuralNetwork/Modules/Output.h"
#include "RecurrentNeuralNetwork/Operations/BackwardOnWeights.h"
#include "RecurrentNeuralNetwork/Operations/backward_on_data.h"
#include "RecurrentNeuralNetwork/Operations/forward.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "gtest/gtest.h"

#include <cudnn.h>

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
using RecurrentNeuralNetwork::Operations::BackwardOnWeights;
using RecurrentNeuralNetwork::Operations::backward_on_data;
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
TEST(BackwardOnWeightsTests, Constructs)
{
  BackwardOnWeights<float, 3> backward {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BackwardOnWeightsTests, Backwards)
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
  h_descriptor.set_strides_by_dimensions();
  h_descriptor.set_descriptor(parameters); 

  WeightSpace weight_space {descriptors};
  WorkAndReserveSpaces spaces {descriptors, x_descriptor};

  Input<float> x {parameters};
  Output<float> y {parameters};
  Hidden<float> hx {parameters};

  BackwardOnWeights<float, 3> backward {};

  const auto result = backward.backward_on_weights(
    descriptors,
    sequence_length_array,
    x_descriptor,
    x,
    h_descriptor,
    hx,
    y_descriptor,
    y,
    weight_space,
    spaces);

  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BackwardOnWeightsTests, BackwardsAfterForwards)
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

  const auto forward_result = forward<float, 3>(
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

  ASSERT_TRUE(forward_result.is_success());

  BackwardOnWeights<float, 3> backward {};

  const auto result = backward.backward_on_weights(
    descriptors,
    sequence_length_array,
    x_descriptor,
    x,
    h_descriptor,
    hx,
    y_descriptor,
    y,
    weight_space,
    spaces);

  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BackwardOnWeightsTests, BackwardsAfterForwardsAndBackwardOnData)
{
  DefaultParameters parameters {};
  SequenceLengthArray sequence_length_array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  sequence_length_array.copy_host_input_to_device(host_array);

  // Descriptors to be used are the same as when used for forward operation.
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
  Input<float> dx {parameters};
  Output<float> y {parameters};
  Output<float> dy {parameters};
  Hidden<float> hx {parameters};
  Hidden<float> hy {parameters};
  Hidden<float> dhy {parameters};
  Hidden<float> dhx {parameters};
  Hidden<float> cx {parameters};
  Hidden<float> cy {parameters};
  Hidden<float> dcy {parameters};
  Hidden<float> dcx {parameters};

  const auto forward_result = forward<float, 3>(
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

  ASSERT_TRUE(forward_result.is_success());

  const auto backward_on_data_result = backward_on_data<float, 3>(
    descriptors,
    sequence_length_array,
    y_descriptor,
    y,
    dy,
    x_descriptor,
    dx,
    h_descriptor,
    hx,
    dhy,
    dhx,
    c_descriptor,
    cx,
    dcy,
    dcx,
    weight_space,
    spaces);

  ASSERT_TRUE(backward_on_data_result.is_success());

  BackwardOnWeights<float, 3> backward {};

  const auto result = backward.backward_on_weights(
    descriptors,
    sequence_length_array,
    x_descriptor,
    x,
    h_descriptor,
    hx,
    y_descriptor,
    y,
    weight_space,
    spaces);

  EXPECT_TRUE(result.is_success());
}

} // namespace Operations
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests