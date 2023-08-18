#include "Descriptor.h"
#include "DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>
#include <stdexcept>

using RecurrentNeuralNetwork::Parameters;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

Descriptor::Descriptor():
  descriptor_{},
  is_set_{false}
{
  HandleUnsuccessfulCuDNNCall create_descriptor {
    "Failed to create RNN descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateRNNDescriptor
  // 7.2.6. cudnnCreateRNNDescriptor(). This function create a generic RNN
  // descriptor object by allocating memory needed to hold its opaque structure.
  create_descriptor(cudnnCreateRNNDescriptor(&descriptor_));

  if (!create_descriptor.is_success())
  {
    throw std::runtime_error(create_descriptor.get_error_message());
  }
}

HandleUnsuccessfulCuDNNCall Descriptor::get_RNN_parameters(
  Parameters& parameters,
  DropoutDescriptor& dropout_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_get_descriptor {
    "Failed to get RNN descriptor"};

  handle_get_descriptor(cudnnGetRNNDescriptor_v8(
    descriptor_,
    &(parameters.algo_),
    &(parameters.cell_mode_),
    &(parameters.bias_mode_),
    &(parameters.direction_mode_),
    &(parameters.input_mode_),
    &(parameters.data_type_),
    &(parameters.math_precision_),
    &(parameters.math_type_),
    &(parameters.input_size_),
    &(parameters.hidden_size_),
    &(parameters.projection_size_),
    &(parameters.number_of_layers_),
    &(dropout_descriptor.descriptor_),
    &(parameters.auxiliary_flags_)
    ));

  return handle_get_descriptor;
}

Descriptor::~Descriptor()
{
  HandleUnsuccessfulCuDNNCall destroy_descriptor {
    "Failed to destroy RNN descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyRNNDescriptor
  destroy_descriptor(cudnnDestroyRNNDescriptor(descriptor_));
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork