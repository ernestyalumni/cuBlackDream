#include "Tensors/ManageDescriptor/SetFor4DTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace Tensors
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor4DTensorTests, DefaultConstructs)
{
  TensorDescriptor descriptor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor4DTensorTests, Destructs)
{
  {
    TensorDescriptor descriptor {};
  }

  SUCCEED();
}

} // namespace Tensors
} // namespace GoogleUnitTests