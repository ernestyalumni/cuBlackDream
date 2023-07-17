#include "Activation/Forward.h"
#include "Activation/ManageDescriptor/ActivationDescriptor.h"
#include "Activation/ManageDescriptor/SetDescriptor.h"
#include "Algebra/Modules/Tensors/Tensor4D.h"
#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "Tensors/ManageDescriptor/SetFor4DTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <vector>

using Activation::Forward;
using Activation::ManageDescriptor::ActivationDescriptor;
using Activation::ManageDescriptor::SetDescriptor;
using Algebra::Modules::Tensors::HostTensor4D;
using Algebra::Modules::Tensors::Tensor4D;
using DeepNeuralNetwork::CuDNNLibraryHandle;
using Tensors::ManageDescriptor::SetFor4DTensor;
using Tensors::ManageDescriptor::TensorDescriptor;
using std::size_t;

namespace GoogleUnitTests
{
namespace Activation
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ForwardTests, Constructs)
{
  Forward forward {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ForwardTests, SetDescriptorSetsForCudnnActivationDescriptorT)
{
  CuDNNLibraryHandle handle {};

  TensorDescriptor x_descriptor {};
  SetFor4DTensor set4d {1, 1, 1, 10, CUDNN_DATA_FLOAT};
  set4d.set_descriptor(x_descriptor);

  HostTensor4D<float> ht {1, 1, 1, 10};
  for (size_t i {0}; i < 10; ++i)
  {
    ht.get(0, 0, 0, i) = static_cast<float>(i);
  }
  Tensor4D<float> x {1, 1, 1, 10};
  x.copy_host_input_to_device(ht);

  ActivationDescriptor activation_descriptor {};
  // S(x) = 1 / (1 + exp(-x))
  SetDescriptor set_descriptor {CUDNN_ACTIVATION_SIGMOID};
  set_descriptor.set_descriptor(activation_descriptor);

  Forward forward {};

  const auto result_status = forward.inplace_activation_forward(
    handle,
    activation_descriptor,
    x_descriptor,
    x);

  EXPECT_TRUE(result_status.is_success());

  HostTensor4D<float> ht_check {1, 1, 1, 10};
  const std::vector<float> empty_values (10, 0.0f);
  auto result_values = ht_check.copy_values(empty_values);
  x.copy_device_to_host(ht_check);

  // It is easy to check these values using Python and numpy:
  // >>> import numpy as np
  // >>> array = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  // >>> np.reciprocal(np.exp(-1 * array) + 1)
  // array([0.5       , 0.73105858, 0.88079708, 0.95257413, 0.98201379,
  //       0.99330715, 0.99752738, 0.99908895, 0.99966465, 0.99987661])

  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 0), 0.5); 
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 1), 0.73105858);
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 2), 0.88079708);
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 3), 0.95257413);
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 4), 0.98201379);
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 5), 0.99330715); 
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 6), 0.99752738);
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 7), 0.99908895);
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 8), 0.99966465); 
  EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, 9), 0.99987661);
}

} // namespace Activation
} // namespace GoogleUnitTests