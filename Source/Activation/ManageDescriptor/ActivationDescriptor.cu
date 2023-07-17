#include "ActivationDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace Activation
{
namespace ManageDescriptor
{

ActivationDescriptor::ActivationDescriptor():
  descriptor_{}
{
  HandleUnsuccessfulCuDNNCall create_descriptor {
    "Failed to create Tensor descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateActivationDescriptor
  // 3.2.6. cudnnCreateActivationDescriptor(). This function create an
  // activation descriptor object by allocating memory needed to hold its opaque
  // structure.
  create_descriptor(cudnnCreateActivationDescriptor(&descriptor_));
}

ActivationDescriptor::~ActivationDescriptor()
{
  HandleUnsuccessfulCuDNNCall destroy_descriptor {
    "Failed to destroy descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyTensorDescriptor
  destroy_descriptor(cudnnDestroyActivationDescriptor(descriptor_));
}

} // namespace ManageDescriptor
} // namespace Activation