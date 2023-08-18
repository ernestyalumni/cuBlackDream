#include "AttentionDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>
#include <stdexcept>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace Transformer
{
namespace ManageDescriptor
{

AttentionDescriptor::AttentionDescriptor():
  descriptor_{}
{
  HandleUnsuccessfulCuDNNCall create_descriptor {
    "Failed to create attention descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateAttnDescriptor
  // 7.2.3. cudnnCreateAttnDescriptor()
  create_descriptor(cudnnCreateAttnDescriptor(&descriptor_));

  if (!create_descriptor.is_success())
  {
    throw std::runtime_error(create_descriptor.get_error_message());
  }
}

AttentionDescriptor::~AttentionDescriptor()
{
  HandleUnsuccessfulCuDNNCall destroy_descriptor {
    "Failed to destroy attention descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyRNNDescriptor
  destroy_descriptor(cudnnDestroyAttnDescriptor(descriptor_));
}

} // namespace ManageDescriptor
} // namespace Transformer