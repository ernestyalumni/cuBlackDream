#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace Tensors
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TensorDescriptorTests, DefaultConstructs)
{
  TensorDescriptor descriptor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TensorDescriptorTests, Destructs)
{
  {
    TensorDescriptor descriptor {};
  }

  SUCCEED();
}

} // namespace ManageDescriptor
} // namespace Tensors
} // namespace GoogleUnitTests