#include "Transformer/ManageDescriptor/AttentionDescriptor.h"
#include "gtest/gtest.h"

using Transformer::ManageDescriptor::AttentionDescriptor;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AttentionDescriptorTests, DefaultConstructs)
{
  AttentionDescriptor attention_descriptor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AttentionDescriptor, Destructs)
{
  {
    AttentionDescriptor attention_descriptor {};

    SUCCEED();
  }
}

} // namespace ManageDescriptor
} // namespace Transformer
} // namespace GoogleUnitTests