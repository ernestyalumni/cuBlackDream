#ifndef UTILITIES_CUDNN_DATA_TYPE_TO_TYPE
#define UTILITIES_CUDNN_DATA_TYPE_TO_TYPE

#include <cstdint>
#include <cudnn.h>

namespace Utilities
{

template <cudnnDataType_t DT>
struct CuDNNDataTypeToType;

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDataType_t
//------------------------------------------------------------------------------

template <>
// CUDNN_DATA_FLOAT - 32-bit single-precision floating-point (float)
struct CuDNNDataTypeToType<CUDNN_DATA_FLOAT>
{
  // We use std::type_identity to map an enumeration value to a specific type.
  using type = std::type_identity<float>;
};

template <>
// CUDNN_DATA_DOUBLE - 64-bit double-precision floating point.
struct CuDNNDataTypeToType<CUDNN_DATA_DOUBLE>
{
  using type = std::type_identity<double>;
};

template <>
// CUDNN_DATA_INT8 - 8-bit signed integer.
struct CuDNNDataTypeToType<CUDNN_DATA_INT8>
{
  using type = std::type_identity<int8_t>;
};

template <>
struct CuDNNDataTypeToType<CUDNN_DATA_INT32>
{
  using type = std::type_identity<int32_t>;
};

template <>
struct CuDNNDataTypeToType<CUDNN_DATA_INT64>
{
  using type = std::type_identity<int64_t>;
};

template <>
struct CuDNNDataTypeToType<CUDNN_DATA_UINT8>
{
  using type = std::type_identity<uint8_t>;
};

template <>
struct CuDNNDataTypeToType<CUDNN_DATA_BOOLEAN>
{
  using type = std::type_identity<bool>;
};

} // namespace Utilities

#endif // UTILITIES_CUDNN_DATA_TYPE_TO_TYPE