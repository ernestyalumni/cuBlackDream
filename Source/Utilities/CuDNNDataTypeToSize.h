#ifndef UTILITIES_CUDNN_DATA_TYPE_TO_SIZE
#define UTILITIES_CUDNN_DATA_TYPE_TO_SIZE

#include <cstddef>
#include <cudnn.h>

namespace Utilities
{

//------------------------------------------------------------------------------
/// \details A function that would map cudnnDataType_t value to its size in
/// run time and not compile time is needed in order to use
/// cudnnGetTensorNdDescriptor, which gets values in a TensorNd descriptor in
/// run time.
//------------------------------------------------------------------------------
std::size_t cuDNN_data_type_to_size(const cudnnDataType_t data_type);

} // namespace Utilities

#endif // UTILITIES_CUDNN_DATA_TYPE_TO_SIZE