#include "RecurrentNeuralNetwork/Parameters.h"
#include "Tensors/ManageDescriptor/SetForNDTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

using RecurrentNeuralNetwork::DefaultParameters;
using Tensors::ManageDescriptor::SetFor3DTensor;
using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace Tensors
{
namespace ManageDescriptor
{

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