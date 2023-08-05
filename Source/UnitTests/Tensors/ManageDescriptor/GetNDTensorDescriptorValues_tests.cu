#include "RecurrentNeuralNetwork/Parameters.h"
#include "Tensors/ManageDescriptor/GetNDTensorDescriptorValues.h"
#include "Tensors/ManageDescriptor/SetForNDTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

#include <array>
#include <cudnn.h>

using RecurrentNeuralNetwork::DefaultParameters;
using Tensors::ManageDescriptor::GetNDTensorDescriptorValues;
using Tensors::ManageDescriptor::SetFor3DTensor;
using Tensors::ManageDescriptor::SetForNDTensor;
using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace Tensors
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetTensorNdDescriptorValuesTests, DefaultConstructs)
{
  GetNDTensorDescriptorValues<6> get_values {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetTensorNdDescriptorValuesTests, GetValuesGetNoValuesOnUnsetDescriptor)
{
  GetNDTensorDescriptorValues<3> get_values {};
  TensorDescriptor h_descriptor {};
  auto result = get_values.get_values(h_descriptor, 3);
   
  EXPECT_TRUE(result.is_success());
  EXPECT_EQ(*get_values.data_type_, 0);
  EXPECT_EQ(get_values.nb_dims_[0], 0);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], -1);
  EXPECT_EQ(get_values.dimensions_array_[1], -1);
  EXPECT_EQ(get_values.dimensions_array_[2], -1);
  EXPECT_EQ(get_values.strides_array_[0], -1);
  EXPECT_EQ(get_values.strides_array_[1], -1);
  EXPECT_EQ(get_values.strides_array_[2], -1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetTensorNdDescriptorValuesTests, GetValuesGetsFor3Dimensions)
{
  GetNDTensorDescriptorValues<3> get_values {};

  TensorDescriptor h_descriptor {};
  DefaultParameters parameters {};
  SetFor3DTensor set3d {parameters};
  const auto set_result = set3d.set_descriptor(h_descriptor, parameters);
  ASSERT_TRUE(set_result.is_success());

  auto result = get_values.get_values(h_descriptor, 3);
  EXPECT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 3);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 2);
  EXPECT_EQ(get_values.dimensions_array_[1], 64);
  EXPECT_EQ(get_values.dimensions_array_[2], 512);
  EXPECT_EQ(get_values.strides_array_[0], 32768);
  EXPECT_EQ(get_values.strides_array_[1], 512);
  EXPECT_EQ(get_values.strides_array_[2], 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetTensorNdDescriptorValuesTests, GetValuesGetsFor4Dimensions)
{
  GetNDTensorDescriptorValues<4> get_values {};
  TensorDescriptor h_descriptor {};
  SetForNDTensor<4> set_tensor {};
  DefaultParameters parameters {};
  parameters.data_type_ = CUDNN_DATA_DOUBLE;
  parameters.math_precision_ = CUDNN_DATA_DOUBLE;

  // e.g. 0 - batch size
  // 1 - 3 layers in RNN
  // 2 - height, 2^12
  // 3 - length 2^11
  std::array<int, 4> input_dimensions_array {60, 3, 4096, 2048};
  set_tensor.set_dimensions(input_dimensions_array);
  set_tensor.set_strides_from_dimensions_as_descending();
  const auto set_result = set_tensor.set_descriptor(h_descriptor, parameters);
  ASSERT_TRUE(set_result.is_success());

  auto result = get_values.get_values(h_descriptor, 4);
  EXPECT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_DOUBLE);
  EXPECT_EQ(get_values.nb_dims_[0], 4);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.nb_dims_[3], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 60);
  EXPECT_EQ(get_values.dimensions_array_[1], 3);
  EXPECT_EQ(get_values.dimensions_array_[2], 4096);
  EXPECT_EQ(get_values.dimensions_array_[3], 2048);
  EXPECT_EQ(get_values.strides_array_[0], 25165824);
  EXPECT_EQ(get_values.strides_array_[1], 8388608);
  EXPECT_EQ(get_values.strides_array_[2], 2048);
  EXPECT_EQ(get_values.strides_array_[3], 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetTensorNdDescriptorValuesTests, GetValuesGetsForLessThan4Dimensions)
{
  GetNDTensorDescriptorValues<4> get_values {};
  TensorDescriptor h_descriptor {};
  SetForNDTensor<4> set_tensor {};
  DefaultParameters parameters {};

  std::array<int, 4> input_dimensions_array {60, 3, 4096, 2048};
  set_tensor.set_dimensions(input_dimensions_array);
  set_tensor.set_strides_from_dimensions_as_descending();
  const auto set_result = set_tensor.set_descriptor(h_descriptor, parameters);

  ASSERT_TRUE(set_result.is_success());

  auto result = get_values.get_values(h_descriptor, 3);
  EXPECT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 4);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.nb_dims_[3], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 60);
  EXPECT_EQ(get_values.dimensions_array_[1], 3);
  EXPECT_EQ(get_values.dimensions_array_[2], 4096);
  EXPECT_EQ(get_values.dimensions_array_[3], -1);
  EXPECT_EQ(get_values.strides_array_[0], 25165824);
  EXPECT_EQ(get_values.strides_array_[1], 8388608);
  EXPECT_EQ(get_values.strides_array_[2], 2048);
  EXPECT_EQ(get_values.strides_array_[3], -1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetTensorNdDescriptorValuesTests, GetValuesGetsForMoreThan4Dimensions)
{
  GetNDTensorDescriptorValues<5> get_values {};
  TensorDescriptor h_descriptor {};
  SetForNDTensor<4> set_tensor {};
  DefaultParameters parameters {};

  std::array<int, 4> input_dimensions_array {60, 3, 4096, 2048};
  set_tensor.set_dimensions(input_dimensions_array);
  set_tensor.set_strides_from_dimensions_as_descending();
  const auto set_result = set_tensor.set_descriptor(h_descriptor, parameters);

  ASSERT_TRUE(set_result.is_success());

  auto result = get_values.get_values(h_descriptor, 5);
  EXPECT_TRUE(result.is_success());

  EXPECT_EQ(*get_values.data_type_, CUDNN_DATA_FLOAT);
  EXPECT_EQ(get_values.nb_dims_[0], 4);
  EXPECT_EQ(get_values.nb_dims_[1], -1);
  EXPECT_EQ(get_values.nb_dims_[2], -1);
  EXPECT_EQ(get_values.nb_dims_[3], -1);
  EXPECT_EQ(get_values.nb_dims_[4], -1);
  EXPECT_EQ(get_values.dimensions_array_[0], 60);
  EXPECT_EQ(get_values.dimensions_array_[1], 3);
  EXPECT_EQ(get_values.dimensions_array_[2], 4096);
  EXPECT_EQ(get_values.dimensions_array_[3], 2048);
  EXPECT_EQ(get_values.dimensions_array_[4], -1);
  EXPECT_EQ(get_values.strides_array_[0], 25165824);
  EXPECT_EQ(get_values.strides_array_[1], 8388608);
  EXPECT_EQ(get_values.strides_array_[2], 2048);
  EXPECT_EQ(get_values.strides_array_[3], 1);
  EXPECT_EQ(get_values.strides_array_[4], -1);
}

} // namespace ManageDescriptor
} // namespace Tensors
} // namespace GoogleUnitTests