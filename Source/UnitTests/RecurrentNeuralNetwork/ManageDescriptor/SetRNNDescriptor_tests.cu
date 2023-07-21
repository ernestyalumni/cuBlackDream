#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/SetDropoutDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/SetRNNDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using DeepNeuralNetwork::CuDNNLibraryHandle;
using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::ManageDescriptor::Descriptor;
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
TEST(CudnnRNNPaddingModeTTests, UnderLyingValues)
{
  //----------------------------------------------------------------------------
  /// \ref 7.1.2.9 cudnnRNNPaddingMode_t
  /// CUDNN_RNN_PADDED_IO_DISABLED - disabled padded input/output.
  //----------------------------------------------------------------------------
  EXPECT_EQ(static_cast<int>(CUDNN_RNN_PADDED_IO_DISABLED), 0);
  EXPECT_EQ(static_cast<int>(CUDNN_RNN_PADDED_IO_ENABLED), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetRNNDescriptorTests, SetDescriptorSets)
{
  DefaultParameters parameters {};

  CuDNNLibraryHandle handle {};
  DropoutDescriptor dropout_descriptor {};
  dropout_descriptor.get_states_size_for_forward(handle);
  SetDropoutDescriptor set_dropout_descriptor {0};
  set_dropout_descriptor.set_descriptor(dropout_descriptor, handle);

  Descriptor descriptor {};

  const auto result =
    ::RecurrentNeuralNetwork::ManageDescriptor::set_rnn_descriptor(
      descriptor,
      parameters,
      dropout_descriptor);

  EXPECT_TRUE(result.is_success());
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests