#include "HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>
#include <iostream> // std::cerr
#include <string>

using std::cerr;

namespace Utilities
{
namespace ErrorHandling
{

HandleUnsuccessfulCuDNNCall::HandleUnsuccessfulCuDNNCall(
  const std::string& error_message
  ):
  error_message_{error_message},
  status_{CUDNN_STATUS_SUCCESS}
{}

void HandleUnsuccessfulCuDNNCall::operator()(
  const cudnnStatus_t cuDNN_status)
{
  status_ = cuDNN_status;

  if (!is_success())
  {
    cerr << error_message_ << " (error code " <<
      // https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-API.pdf
      // 3.2.45 cudnn_ops_infer.so Library
      // This function converts the cuDNN status code to a NULL terminated
      // (ASCIIZ) static string.
      cudnnGetErrorString(status_) << ")!\n";
  }
}

} // namespace ErrorHandling
} // namespace Utilities