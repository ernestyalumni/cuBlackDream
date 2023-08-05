#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/Modules/GetWeightsAndBias.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "Tensors/ManageDescriptor/GetNDTensorDescriptorValues.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::Modules::GetWeightsAndBias;
using RecurrentNeuralNetwork::WeightSpace;
using Tensors::ManageDescriptor::GetNDTensorDescriptorValues;
using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace Modules
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetWeightsAndBiasTests, Constructs)
{
  DefaultParameters parameters {};

  GetWeightsAndBias get_weight_and_bias {parameters};

  EXPECT_EQ(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_EQ(get_weight_and_bias.bias_address_, nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetWeightsAndBiasTests, GetWeightsAndBiasWorks)
{
  DefaultParameters parameters {};
  GetWeightsAndBias get_weight_and_bias {parameters};
  LibraryHandleDropoutRNN descriptors {parameters};
  WeightSpace weight_space {descriptors};
  
  TensorDescriptor weight_descriptor {};
  TensorDescriptor bias_descriptor {};

  auto result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    0,
    weight_space,
    0,
    weight_descriptor,
    bias_descriptor);

  EXPECT_TRUE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  GetNDTensorDescriptorValues<3> get_values {};
  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetWeightsAndBiasTests, GetWeightsAndBiasWorksOnMultipleLayers)
{
  DefaultParameters parameters {};
  // We will demonstrate that pseudoLayers only goes up to number_of_layers_ for
  // this setting in DefaultParameters, CUDNN_UNIDIRECTIONAL and CUDNN_RNN_RELU.
  parameters.number_of_layers_ = 3;
  GetWeightsAndBias get_weight_and_bias {parameters};
  LibraryHandleDropoutRNN descriptors {parameters};
  WeightSpace weight_space {descriptors};
  
  TensorDescriptor weight_descriptor {};
  TensorDescriptor bias_descriptor {};

  // For unidirectional RNNs,
  // pseudoLayer = 1 is the first hidden layer.
  // pseudoLayer = 0 was the RNN input layer.

  auto result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    1,
    weight_space,
    0,
    weight_descriptor,
    bias_descriptor);

  EXPECT_TRUE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  GetNDTensorDescriptorValues<3> get_values {};
  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  // Second hidden layer.
  result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    2,
    weight_space,
    0,
    weight_descriptor,
    bias_descriptor);

  EXPECT_TRUE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  // No more hidden layers past number_of_layers.
  result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    3,
    weight_space,
    0,
    weight_descriptor,
    bias_descriptor);

  EXPECT_FALSE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  // We expect to get the previous values.
  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetWeightsAndBiasTests, GetWeightsAndBiasLimitedByNumberOfLayersSpecified)
{
  // We confirm that the number_of_layers (e.g. = 2 in this case) limits the
  // number of pseudoLayers available.
  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDescriptor_v8
  // Recall taht numLayers or number_of_layers is an input and is the number of
  // stacked, physical layers in the deep RNN model. 
  DefaultParameters parameters {};
  GetWeightsAndBias get_weight_and_bias {parameters};
  LibraryHandleDropoutRNN descriptors {parameters};
  WeightSpace weight_space {descriptors};
  
  TensorDescriptor weight_descriptor {};
  TensorDescriptor bias_descriptor {};

  // From CUDA API, for unidirectional RNNs,
  // pseudoLayer = 1 is the first hidden layer.
  // pseudoLayer = 0 was the RNN input layer, but
  // EY (20230806) If I specified

  auto result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    1,
    weight_space,
    0,
    weight_descriptor,
    bias_descriptor);

  EXPECT_TRUE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  GetNDTensorDescriptorValues<3> get_values {};
  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  // No more.
  result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    2,
    weight_space,
    0,
    weight_descriptor,
    bias_descriptor);

  EXPECT_FALSE(result.is_success());

  // We expect to get the previous values.

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetWeightsAndBiasTests, OutOfLimitLinearLayerIDGivesNullptr)
{
  DefaultParameters parameters {};
  GetWeightsAndBias get_weight_and_bias {parameters};
  LibraryHandleDropoutRNN descriptors {parameters};
  WeightSpace weight_space {descriptors};
  
  TensorDescriptor weight_descriptor {};
  TensorDescriptor bias_descriptor {};

  auto result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    0,
    weight_space,
    2,
    weight_descriptor,
    bias_descriptor);

  EXPECT_FALSE(result.is_success());

  EXPECT_EQ(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_EQ(get_weight_and_bias.bias_address_, nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetWeightsAndBiasTests, LinearLayerIDOf1ForCUDNN_RNN_RELU)
{
  DefaultParameters parameters {};
  GetWeightsAndBias get_weight_and_bias {parameters};
  LibraryHandleDropoutRNN descriptors {parameters};
  WeightSpace weight_space {descriptors};
  
  TensorDescriptor weight_descriptor {};
  TensorDescriptor bias_descriptor {};

  auto result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    0,
    weight_space,
    1,
    weight_descriptor,
    bias_descriptor);

  EXPECT_TRUE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  GetNDTensorDescriptorValues<3> get_values {};
  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    1,
    weight_space,
    1,
    weight_descriptor,
    bias_descriptor);

  EXPECT_TRUE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  // No more hidden layers.

  result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    2,
    weight_space,
    1,
    weight_descriptor,
    bias_descriptor);

  EXPECT_FALSE(result.is_success());

  EXPECT_NE(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_NE(get_weight_and_bias.bias_address_, nullptr);

  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 262144);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  // No higher linear ID because only values 0, 1 allowed for CUDNN_RNN_RELU.

  result = get_weight_and_bias.get_weight_and_bias(
    descriptors,
    1,
    weight_space,
    2,
    weight_descriptor,
    bias_descriptor);

  EXPECT_FALSE(result.is_success());

  EXPECT_EQ(get_weight_and_bias.weight_matrix_address_, nullptr);
  EXPECT_EQ(get_weight_and_bias.bias_address_, nullptr);

  result = get_values.get_values(weight_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 0);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);

  result = get_values.get_values(bias_descriptor, 3);
  ASSERT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 0);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 1);
  EXPECT_EQ(get_values.dimensions_array_[1], 512);
  EXPECT_EQ(get_values.dimensions_array_[2], 1);
  EXPECT_EQ(get_values.strides_array_[0], 512);
  EXPECT_EQ(get_values.strides_array_[1], 1);
  EXPECT_EQ(get_values.strides_array_[2], 1);
}

} // namespace Modules
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests