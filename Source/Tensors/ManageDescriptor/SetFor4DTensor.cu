#include "SetFor4DTensor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace Tensors
{
namespace ManageDescriptor
{

SetFor4DTensor::SetFor4DTensor(
  const std::size_t n,
  const std::size_t c,
  const std::size_t h,
  const std::size_t w,
  const cudnnDataType_t data_type,
  const cudnnTensorFormat_t format
  ):
  n_{static_cast<int>(n)},
  c_{static_cast<int>(c)},
  h_{static_cast<int>(h)},
  w_{static_cast<int>(w)},
  number_of_elements_{n_ * c_ * h_ * w_},
  data_type_{data_type},
  format_{format}  
{}

HandleUnsuccessfulCuDNNCall SetFor4DTensor::set_descriptor(
  cudnnTensorDescriptor_t& tensor_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_set_descriptor {
    "Failed to set Tensor 4D descriptor"};

  handle_set_descriptor(cudnnSetTensor4dDescriptor(
    tensor_descriptor,
    format_,
    data_type_,
    n_,
    c_,
    h_,
    w_));

  return handle_set_descriptor;
}

} // namespace ManageDescriptor
} // namespace Tensors