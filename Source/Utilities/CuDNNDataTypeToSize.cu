#include "CuDNNDataTypeToSize.h"

#include <cstddef>
#include <cstdint>
#include <cudnn.h>

namespace Utilities
{

std::size_t cuDNN_data_type_to_size(const cudnnDataType_t data_type)
{
  switch (data_type)
  {
    case CUDNN_DATA_FLOAT:
      return sizeof(float);
    case CUDNN_DATA_DOUBLE:
      return sizeof(double);
    case CUDNN_DATA_INT8:
      return sizeof(int8_t);
    case CUDNN_DATA_INT32:
      return sizeof(int32_t);
    case CUDNN_DATA_INT64:
      return sizeof(int64_t);
    case CUDNN_DATA_UINT8:
      return sizeof(uint8_t);
    case CUDNN_DATA_BOOLEAN:
      return sizeof(bool);
  }

  return 0;
}

} // namespace Utilities