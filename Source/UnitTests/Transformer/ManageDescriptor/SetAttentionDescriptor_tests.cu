#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/SetDropoutDescriptor.h"
#include "Transformer/ManageDescriptor/AttentionDescriptor.h"
#include "Transformer/ManageDescriptor/SetAttentionDescriptor.h"
#include "UnitTests/Transformer/TestValues.h"
#include "gtest/gtest.h"

using DeepNeuralNetwork::CuDNNLibraryHandle;
using GoogleUnitTests::Transformer::ExampleParameters;
using RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::SetDropoutDescriptor;
using Transformer::Attention::Parameters;
using Transformer::ManageDescriptor::AttentionDescriptor;
using Transformer::ManageDescriptor::set_attention_descriptor;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetAttentionDescriptorTests, SetsDescriptor)
{
  CuDNNLibraryHandle handle {};
  DropoutDescriptor attention_dropout_descriptor {};
  attention_dropout_descriptor.get_states_size_for_forward(handle);
  DropoutDescriptor post_dropout_descriptor {};
  post_dropout_descriptor.get_states_size_for_forward(handle);
  ExampleParameters parameters {};
  AttentionDescriptor attention_descriptor {};

  const auto result = set_attention_descriptor(
    attention_descriptor,
    parameters,
    attention_dropout_descriptor,
    post_dropout_descriptor);

  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetAttentionDescriptorTests, SetsDescriptorAfterSettingDropout)
{
  CuDNNLibraryHandle handle {};
  DropoutDescriptor attention_dropout_descriptor {};
  attention_dropout_descriptor.get_states_size_for_forward(handle);
  DropoutDescriptor post_dropout_descriptor {};
  post_dropout_descriptor.get_states_size_for_forward(handle);
  ExampleParameters parameters {};
  AttentionDescriptor attention_descriptor {};

  SetDropoutDescriptor set_dropout_descriptor {0.1};
  const auto attention_result = set_dropout_descriptor.set_descriptor(
    attention_dropout_descriptor,
    handle);
  ASSERT_TRUE(attention_result.is_success());
  set_dropout_descriptor.dropout_ = 0.2;
  set_dropout_descriptor.seed_ = 1327ull;
  const auto post_result = set_dropout_descriptor.set_descriptor(
    post_dropout_descriptor,
    handle);

  ASSERT_TRUE(post_result.is_success());

  const auto result = set_attention_descriptor(
    attention_descriptor,
    parameters,
    attention_dropout_descriptor,
    post_dropout_descriptor);

  EXPECT_TRUE(result.is_success());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetAttentionDescriptorTests, GetsParameters)
{
  CuDNNLibraryHandle handle {};

  // Test setup.
  AttentionDescriptor attention_descriptor {};

  DropoutDescriptor attention_dropout_descriptor {};
  attention_dropout_descriptor.get_states_size_for_forward(handle);
  DropoutDescriptor post_dropout_descriptor {};
  post_dropout_descriptor.get_states_size_for_forward(handle);
  ExampleParameters parameters {};

  const auto set_result = set_attention_descriptor(
    attention_descriptor,
    parameters,
    attention_dropout_descriptor,
    post_dropout_descriptor);
  ASSERT_TRUE(set_result.is_success());

  // Make values different from what's expected above.
  Parameters result_parameters {
    0,
    3,
    1.5,
    CUDNN_DATA_HALF,
    CUDNN_DATA_HALF,
    CUDNN_DEFAULT_MATH,
    777,
    666,
    444,
    333,
    333,
    111,
    55,
    444,
    222,
    111,
    2};

  DropoutDescriptor result_attention_dropout_descriptor {};
  DropoutDescriptor result_post_dropout_descriptor {};
  result_attention_dropout_descriptor.get_states_size_for_forward(handle);
  result_post_dropout_descriptor.get_states_size_for_forward(handle);

  /* 
  const auto result = get_attention_descriptor(
    attention_descriptor,
    result_parameters,
    result_attention_dropout_descriptor,
    result_post_dropout_descriptor);

  EXPECT_TRUE(result.is_success());
  }
  */
}

} // namespace ManageDescriptor
} // namespace Transformer
} // namespace GoogleUnitTests