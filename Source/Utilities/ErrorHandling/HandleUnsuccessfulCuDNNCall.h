#ifndef UTILITIES_HANDLE_UNSUCCESSFUL_CUDNN_CALL_H
#define UTILITIES_HANDLE_UNSUCCESSFUL_CUDNN_CALL_H

#include <cudnn.h>
#include <string>

namespace Utilities
{
namespace ErrorHandling
{

class HandleUnsuccessfulCuDNNCall
{
  public:

    inline static const std::string default_error_message_ {
      "cuDNN status Success was not returned."};

    HandleUnsuccessfulCuDNNCall(
      const std::string& error_message = default_error_message_);

    ~HandleUnsuccessfulCuDNNCall() = default;

    inline bool is_success() const
    {
      return status_ == CUDNN_STATUS_SUCCESS;
    }

    void operator()(const cudnnStatus_t cuDNN_status);

    cudnnStatus_t get_status() const
    {
      return status_;
    }

  private:

    std::string error_message_;

    cudnnStatus_t status_;
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_HANDLE_UNSUCCESSFUL_CUDNN_CALL_H
