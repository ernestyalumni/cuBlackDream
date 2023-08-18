#include "Transformer/Attention/WeightWorkAndReserveSpaces.h"
#include "Transformer/ManageDescriptor/LibraryHandleDropoutsAttention.h"
#include "UnitTests/Transformer/TestValues.h"
#include "gtest/gtest.h"

using GoogleUnitTests::Transformer::ExampleParameters;
using Transformer::ManageDescriptor::LibraryHandleDropoutsAttention;
using Transformer::Attention::WeightWorkAndReserveSpaces;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WeightWorkAndReserveSpacesTests, Constructs)
{
  ExampleParameters parameters {};
  LibraryHandleDropoutsAttention descriptors {parameters};

  WeightWorkAndReserveSpaces spaces {descriptors};

  EXPECT_EQ(spaces.get_weight_space_size(), 411808);
  EXPECT_EQ(spaces.get_work_space_size(), 0);
  EXPECT_EQ(spaces.get_reserve_space_size(), 12133120);
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests