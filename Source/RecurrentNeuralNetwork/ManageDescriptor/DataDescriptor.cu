#include "DataDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>
#include <stdexcept>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

DataDescriptor::DataDescriptor():
  descriptor_{}
{
  HandleUnsuccessfulCuDNNCall create_descriptor {
    "Failed to create RNN Data descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateRNNDescriptor
  // 7.2.6. cudnnCreateRNNDescriptor(). This function create a generic RNN
  // descriptor object by allocating memory needed to hold its opaque structure.
  create_descriptor(cudnnCreateRNNDataDescriptor(&descriptor_));

  if (!create_descriptor.is_success())
  {
    throw std::runtime_error(create_descriptor.get_error_message());
  }
}

DataDescriptor::~DataDescriptor()
{
  HandleUnsuccessfulCuDNNCall destroy_descriptor {
    "Failed to destroy RNN Data descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyRNNDescriptor
  destroy_descriptor(cudnnDestroyRNNDataDescriptor(descriptor_));
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork