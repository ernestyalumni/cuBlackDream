#include "Activation/ManageDescriptor/ActivationDescriptor.h"
#include "Activation/ManageDescriptor/SetDescriptor.h"
#include "gtest/gtest.h"

using Activation::ManageDescriptor::ActivationDescriptor;
using Activation::ManageDescriptor::SetDescriptor;

namespace GoogleUnitTests
{
namespace Activation
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDescriptorTests, Constructs)
{
  SetDescriptor set_descriptor {CUDNN_ACTIVATION_SIGMOID};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDescriptorTests, SetDescriptorSetsForCudnnActivationDescriptorT)
{
  ActivationDescriptor descriptor {};

  SetDescriptor set_descriptor {CUDNN_ACTIVATION_SIGMOID};

  set_descriptor.set_descriptor(descriptor.descriptor_);

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetDescriptorTests, SetDescriptorSetsForActivationDescriptor)
{
  ActivationDescriptor descriptor {};

  SetDescriptor set_descriptor {CUDNN_ACTIVATION_SIGMOID};

  set_descriptor.set_descriptor(descriptor);

  SUCCEED();
}

} // namespace ManageDescriptor
} // namespace Activation
} // namespace GoogleUnitTests