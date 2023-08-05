#include "RunTimeVoidPtrArray.h"

#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cstddef>
#include <cuda_runtime.h> // cudaFree, cudaMalloc, cudaMemcpyAsync

using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

RunTimeVoidPtrArray::RunTimeVoidPtrArray():
  values_{nullptr},
  total_size_{0}
{}

RunTimeVoidPtrArray::~RunTimeVoidPtrArray()
{
  if (values_ != nullptr)
  {
    HandleUnsuccessfulCUDACall handle_free {"Failed to free device array"};
    handle_free(cudaFree(values_));
  }
}

HandleUnsuccessfulCUDACall RunTimeVoidPtrArray::initialize(
  const std::size_t total_size)
{
  HandleUnsuccessfulCUDACall handle_malloc {"Failed to allocate device array"};
  handle_malloc(cudaMalloc(&values_, total_size));

  if (handle_malloc.is_cuda_success())
  {
    total_size_ = total_size;
  }

  return handle_malloc;
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
