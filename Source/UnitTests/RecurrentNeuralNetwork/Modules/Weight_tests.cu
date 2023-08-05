#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/Modules/GetWeightsAndBias.h"
#include "RecurrentNeuralNetwork/Modules/Weight.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::Modules::GetWeightsAndBias;
using RecurrentNeuralNetwork::Modules::Weight;
using RecurrentNeuralNetwork::WeightSpace;
using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace Modules
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WeightTests, ConstructFromTensorDescriptor)
{
  DefaultParameters parameters {};
  parameters.hidden_size_ = 600;
  parameters.projection_size_ = 600;
  parameters.number_of_layers_ = 3;
  LibraryHandleDropoutRNN descriptors {parameters};
  GetWeightsAndBias get_weight_and_bias {parameters};
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

  ASSERT_TRUE(result.is_success());

  Weight weight {weight_descriptor};

  EXPECT_EQ(weight.number_of_rows_, 600);
  EXPECT_EQ(weight.number_of_columns_, 512);
  EXPECT_EQ(weight.data_type_, CUDNN_DATA_FLOAT);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WeightTests, CopiesFromWeightAddress)
{
  DefaultParameters parameters {};
  parameters.hidden_size_ = 600;
  parameters.projection_size_ = 600;
  parameters.number_of_layers_ = 3;
  LibraryHandleDropoutRNN descriptors {parameters};
  GetWeightsAndBias get_weight_and_bias {parameters};
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

  ASSERT_TRUE(result.is_success());

  Weight weight {weight_descriptor};

  auto cuda_result = weight.copy_from(get_weight_and_bias);

  EXPECT_TRUE(cuda_result.is_cuda_success());
}


} // namespace Modules
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests