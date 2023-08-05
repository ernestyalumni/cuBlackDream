#include "Utilities/CuDNNDataTypeToType.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cudnn.h>
#include <type_traits>

using Utilities::CuDNNDataTypeToType;

namespace GoogleUnitTests
{
namespace Utilities
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuDNNDataTypeToType, CUDNNDATAFLOATYieldsFloatType)
{
  EXPECT_TRUE(
    (std::is_same_v<CuDNNDataTypeToType<CUDNN_DATA_FLOAT>::type::type, float>));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuDNNDataTypeToType, CUDNNDATADOUBLEYieldsDoubleType)
{
  EXPECT_TRUE(
    (std::is_same_v<CuDNNDataTypeToType<CUDNN_DATA_DOUBLE>::type::type, double>)
  );
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuDNNDataTypeToType, CUDNNDATAINT8YieldsInt8Type)
{
  EXPECT_TRUE(
    (std::is_same_v<CuDNNDataTypeToType<CUDNN_DATA_INT8>::type::type, int8_t>));
}

} // namespace Utilities
} // namespace GoogleUnitTests