#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/SetDropoutDescriptor.h"
#include "gtest/gtest.h"

using DeepNeuralNetwork::CuDNNLibraryHandle;
using RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::SetDropoutDescriptor;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDropoutDescriptor, Constructs)
{
  SetDropoutDescriptor set_dropout_descriptor {42};
  EXPECT_FLOAT_EQ(set_dropout_descriptor.dropout_, 42);

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDropoutDescriptor, SetDescriptorSets)
{
  DropoutDescriptor dropout_descriptor {};
  CuDNNLibraryHandle handle {};
  dropout_descriptor.get_states_size_for_forward(handle);

  SetDropoutDescriptor set_dropout_descriptor {0};
  const auto result =
    set_dropout_descriptor.set_descriptor(dropout_descriptor, handle);

  EXPECT_TRUE(result.is_success());
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests