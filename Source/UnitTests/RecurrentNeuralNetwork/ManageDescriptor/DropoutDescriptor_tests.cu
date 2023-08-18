#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "gtest/gtest.h"

using DeepNeuralNetwork::CuDNNLibraryHandle;
using RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DropoutDescriptor, DefaultConstructs)
{
  DropoutDescriptor dropout_descriptor {};
  EXPECT_FALSE(dropout_descriptor.is_states_size_known_);
  EXPECT_TRUE(dropout_descriptor.states_ == nullptr);
  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DropoutDescriptor, Destructs)
{
  {
    DropoutDescriptor dropout_descriptor {};
  }
  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DropoutDescriptor, GetsStates)
{
  DropoutDescriptor dropout_descriptor {};
  CuDNNLibraryHandle handle {};

  const auto result = dropout_descriptor.get_states_size_for_forward(handle);

  EXPECT_TRUE(result.is_success());
  EXPECT_TRUE(dropout_descriptor.is_states_size_known_);
  EXPECT_TRUE(dropout_descriptor.states_ != nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DropoutDescriptor, DestructsAfterGetsStates)
{
  {
    DropoutDescriptor dropout_descriptor {};
    CuDNNLibraryHandle handle {};

    const auto result = dropout_descriptor.get_states_size_for_forward(handle);

    ASSERT_TRUE(result.is_success());
    ASSERT_TRUE(dropout_descriptor.is_states_size_known_);
  }

  SUCCEED();
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests