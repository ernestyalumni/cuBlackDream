#include "SequenceDataDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>
#include <stdexcept>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace Transformer
{
namespace ManageDescriptor
{

SequenceDataDescriptor::SequenceDataDescriptor():
  descriptor_{}
{
  HandleUnsuccessfulCuDNNCall create_descriptor {
    "Failed to create sequence data descriptor"};

  create_descriptor(cudnnCreateSeqDataDescriptor(&descriptor_));

  if (!create_descriptor.is_success())
  {
    throw std::runtime_error(create_descriptor.get_error_message());
  }
}

SequenceDataDescriptor::~SequenceDataDescriptor()
{
  HandleUnsuccessfulCuDNNCall destroy_descriptor {
    "Failed to destroy sequence data descriptor"};

  destroy_descriptor(cudnnDestroySeqDataDescriptor(descriptor_));
}

} // namespace ManageDescriptor
} // namespace Transformer