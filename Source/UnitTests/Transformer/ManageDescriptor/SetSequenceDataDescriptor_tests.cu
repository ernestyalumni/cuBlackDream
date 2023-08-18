#include "Transformer/ManageDescriptor/SequenceDataDescriptor.h"
#include "Transformer/ManageDescriptor/SetSequenceDataDescriptor.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using Transformer::ManageDescriptor::SequenceDataDescriptor;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
/// \ref 7.2.53. cudnnSetSeqDataDescriptor()
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqDataDescriptor
//------------------------------------------------------------------------------
TEST(SetSequenceDescriptorTests, UniqueIdentifiersForAll4Dimensions)
{
  EXPECT_EQ(CUDNN_SEQDATA_TIME_DIM, 0);
  EXPECT_EQ(CUDNN_SEQDATA_BATCH_DIM, 1);
  EXPECT_EQ(CUDNN_SEQDATA_BEAM_DIM, 2);
  EXPECT_EQ(CUDNN_SEQDATA_VECT_DIM, 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetSequenceDescriptorTests, SetsDescriptor)
{
  SequenceDataDescriptor descriptor {};
}

} // namespace ManageDescriptor
} // namespace Transformer
} // namespace GoogleUnitTests