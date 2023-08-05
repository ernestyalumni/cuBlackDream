#include "RecurrentNeuralNetwork/Parameters.h"
#include "Tensors/ManageDescriptor/SetForNDTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

#include <array>

using RecurrentNeuralNetwork::DefaultParameters;
using Tensors::ManageDescriptor::SetFor3DTensor;
using Tensors::ManageDescriptor::SetForNDTensor;
using Tensors::ManageDescriptor::TensorDescriptor;
using std::array;

namespace GoogleUnitTests
{
namespace Tensors
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetForNDTensorTests, DefaultConstructs)
{
  SetForNDTensor<4> set_tensor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetForNDTensorTests, SetsDimensionsArrayFromArray)
{
  SetForNDTensor<4> set_tensor {};

  // e.g. 0 - batch size
  // 1 - 3 layers in RNN * 2 for bidirectional.
  // 2 - height, 2^12 
  // 3 - length 2^11
  array<int, 4> input_dimensions {30, 3 * 2, 4096, 2048};

  set_tensor.set_dimensions(input_dimensions);

  EXPECT_EQ(set_tensor.get_dimensions_array_value(0), 30);
  EXPECT_EQ(set_tensor.get_dimensions_array_value(1), 6);
  EXPECT_EQ(set_tensor.get_dimensions_array_value(2), 4096);
  EXPECT_EQ(set_tensor.get_dimensions_array_value(3), 2048);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetForNDTensorTests, SetsStridesArrayFromArray)
{
  SetForNDTensor<5> set_tensor {};

  // e.g. 0 - batch size
  // 1 - 3 layers in RNN * 2 for bidirectional.
  // 2 - height, 2^12 
  // 3 - length 2^11
  // 4 - depth 2^10
  array<int, 5> input_dimensions {30, 3 * 2, 4096, 2048, 1024};

  set_tensor.set_strides(input_dimensions);

  EXPECT_EQ(set_tensor.get_strides_array_value(0), 30);
  EXPECT_EQ(set_tensor.get_strides_array_value(1), 6);
  EXPECT_EQ(set_tensor.get_strides_array_value(2), 4096);
  EXPECT_EQ(set_tensor.get_strides_array_value(3), 2048);
  EXPECT_EQ(set_tensor.get_strides_array_value(4), 1024);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetForNDTensorTests, SetsStridesFromDimensionsAsDescending)
{
  SetForNDTensor<4> set_tensor {};
  array<int, 4> input_dimensions {2, 3, 256, 128};

  set_tensor.set_dimensions(input_dimensions);
  set_tensor.set_strides_from_dimensions_as_descending();

  EXPECT_EQ(set_tensor.get_strides_array_value(0), 98304);
  EXPECT_EQ(set_tensor.get_strides_array_value(1), 32768);
  EXPECT_EQ(set_tensor.get_strides_array_value(2), 128);
  EXPECT_EQ(set_tensor.get_strides_array_value(3), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor3DTensorTests, DefaultConstructs)
{
  SetFor3DTensor set3d {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor3DTensorTests, SetFromParametersForHiddenLayersSets)
{
  DefaultParameters parameters {};

  SetFor3DTensor set3d {};

  set3d.set_for_hidden_layers(parameters);

  EXPECT_EQ(set3d.get_dimensions_array_value(0), 2);
  EXPECT_EQ(set3d.get_dimensions_array_value(1), 64);
  EXPECT_EQ(set3d.get_dimensions_array_value(2), 512);

  EXPECT_EQ(set3d.get_strides_array_value(0), 32768);
  EXPECT_EQ(set3d.get_strides_array_value(1), 512);
  EXPECT_EQ(set3d.get_strides_array_value(2), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor3DTensorTests, ConstructsWithParameters)
{
  DefaultParameters parameters {};
  SetFor3DTensor set3d {parameters};

  EXPECT_EQ(set3d.get_dimensions_array_value(0), 2);
  EXPECT_EQ(set3d.get_dimensions_array_value(1), 64);
  EXPECT_EQ(set3d.get_dimensions_array_value(2), 512);

  EXPECT_EQ(set3d.get_strides_array_value(0), 32768);
  EXPECT_EQ(set3d.get_strides_array_value(1), 512);
  EXPECT_EQ(set3d.get_strides_array_value(2), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor3DTensorTests, SetDescriptorSets)
{
  TensorDescriptor h_descriptor {};
  DefaultParameters parameters {};
  SetFor3DTensor set3d {parameters};

  const auto result = set3d.set_descriptor(h_descriptor, parameters);

  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor3DTensorTests, SetDescriptorSetsForTwo)
{
  TensorDescriptor h_descriptor {};
  TensorDescriptor c_descriptor {};

  DefaultParameters parameters {};
  SetFor3DTensor set3d {parameters};

  const auto result1 = set3d.set_descriptor(h_descriptor, parameters);
  const auto result2 = set3d.set_descriptor(c_descriptor, parameters);

  EXPECT_TRUE(result1.is_success());
  EXPECT_TRUE(result2.is_success());
}

} // namespace ManageDescriptor
} // namespace Tensors
} // namespace GoogleUnitTests