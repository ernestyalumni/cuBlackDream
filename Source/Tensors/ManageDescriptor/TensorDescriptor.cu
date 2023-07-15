#include "TensorDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace Tensors
{
namespace ManageDescriptor
{

TensorDescriptor::TensorDescriptor():
  descriptor_{}
{
  HandleUnsuccessfulCuDNNCall create_descriptor {
    "Failed to create Tensor descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateTensorDescriptor
  // 3.2.16. cudnnCreateTensorDescriptor(). This function create a generic
  // tensor descriptor object by allocating memory needed to hold its opaque
  // structure. Data is initialized to all zeros.
  create_descriptor(cudnnCreateTensorDescriptor(&descriptor_));
}

TensorDescriptor::~TensorDescriptor()
{
  HandleUnsuccessfulCuDNNCall destroy_descriptor {
    "Failed to destroy descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyTensorDescriptor
  destroy_descriptor(cudnnDestroyTensorDescriptor(descriptor_));
}

} // namespace ManageDescriptor
} // namespace Tensors