#include "Transformer/ManageDescriptor/LibraryHandleDropoutsAttention.h"
#include "UnitTests/Transformer/TestValues.h"
#include "gtest/gtest.h"

using GoogleUnitTests::Transformer::ExampleParameters;
using Transformer::ManageDescriptor::LibraryHandleDropoutsAttention;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LibraryHandleDropoutsAttentionTests, Constructs)
{
  ExampleParameters parameters {};

  LibraryHandleDropoutsAttention descriptors {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LibraryHandleDropoutsAttentionTests, Destructs)
{
  {
    ExampleParameters parameters {};

    LibraryHandleDropoutsAttention descriptors {parameters};
  }

  SUCCEED();
}

} // namespace ManageDescriptor
} // namespace Transformer
} // namespace GoogleUnitTests