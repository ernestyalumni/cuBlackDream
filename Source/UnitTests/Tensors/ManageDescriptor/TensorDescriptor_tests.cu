#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace Tensors
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

} // namespace Tensors
} // namespace GoogleUnitTests