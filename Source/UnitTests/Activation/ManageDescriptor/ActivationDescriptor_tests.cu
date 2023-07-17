#include "Activation/ManageDescriptor/ActivationDescriptor.h"
#include "gtest/gtest.h"

using Activation::ManageDescriptor::ActivationDescriptor;

namespace GoogleUnitTests
{
namespace Activation
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ActivationDescriptorTests, DefaultConstructs)
{
  ActivationDescriptor descriptor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ActivationDescriptorTests, Destructs)
{
  {
    ActivationDescriptor descriptor {};
  }

  SUCCEED();
}

} // namespace ManageDescriptor
} // namespace Activation
} // namespace GoogleUnitTests