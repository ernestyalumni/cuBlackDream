#include "ActivationDescriptor.h"
#include "SetDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace Activation
{

namespace ManageDescriptor
{

SetDescriptor::SetDescriptor(
  const cudnnActivationMode_t mode,
  const double clipping_threshold,
  const cudnnNanPropagation_t option
  ):
  clipping_threshold_{clipping_threshold},
  mode_{mode},
  option_{option}
{}

HandleUnsuccessfulCuDNNCall SetDescriptor::set_descriptor(
  cudnnActivationDescriptor_t& activation_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_set_descriptor {
    "Failed to set activation descriptor"};

  handle_set_descriptor(cudnnSetActivationDescriptor(
    activation_descriptor,
    mode_,
    option_,
    clipping_threshold_));

  return handle_set_descriptor;
}

HandleUnsuccessfulCuDNNCall SetDescriptor::set_descriptor(
  ActivationDescriptor& activation_descriptor)
{
  return set_descriptor(activation_descriptor.descriptor_);
}

} // namespace ManageDescriptor
} // namespace Activation
