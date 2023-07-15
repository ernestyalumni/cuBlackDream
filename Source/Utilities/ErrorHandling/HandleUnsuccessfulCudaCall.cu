#include "HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h>
#include <iostream> // std::cerr
#include <string>

using std::cerr;

namespace Utilities
{
namespace ErrorHandling
{

HandleUnsuccessfulCUDACall::HandleUnsuccessfulCUDACall(
  const std::string& error_message
  ):
  error_message_{error_message},
  cuda_error_{cudaSuccess}
{}

void HandleUnsuccessfulCUDACall::operator()(const cudaError_t cuda_error)
{
  cuda_error_ = cuda_error;

  if (!is_cuda_success())
  {
    cerr << error_message_ << " (error code " <<
      cudaGetErrorString(cuda_error_) << ")!\n";
  }
}

} // namespace ErrorHandling
} // namespace Utilities