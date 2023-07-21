#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "Descriptor.h"
#include "DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "SetRNNDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

HandleUnsuccessfulCuDNNCall set_rnn_descriptor(
  Descriptor& descriptor,
  const RecurrentNeuralNetwork::Parameters& parameters,
  DropoutDescriptor& dropout_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_set_descriptor {
    "Failed to set RNN descriptor"};

  if (dropout_descriptor.is_states_size_known_)
  {
    handle_set_descriptor(cudnnSetRNNDescriptor_v8(
      descriptor.descriptor_,
      parameters.algo_,
      parameters.cell_mode_,
      parameters.bias_mode_,
      parameters.direction_mode_,
      parameters.input_mode_,         
      parameters.data_type_,
      parameters.math_precision_,
      parameters.math_type_,
      parameters.input_size_,
      parameters.hidden_size_,
      parameters.projection_size_,
      parameters.number_of_layers_,
      dropout_descriptor.descriptor_,
      parameters.auxiliary_flags_));

    if (handle_set_descriptor.is_success())
    {
      descriptor.is_set_ = true;
    }

    return handle_set_descriptor;
  }
  else
  {
    handle_set_descriptor(CUDNN_STATUS_NOT_INITIALIZED);

    return handle_set_descriptor;
  }
}

bool build_with_NVRTC_if_dynamic(
  DeepNeuralNetwork::CuDNNLibraryHandle& handle,
  Descriptor& descriptor,
  const RecurrentNeuralNetwork::Parameters& parameters)
{
  HandleUnsuccessfulCuDNNCall handle_build_dynamic {
    "Failed to compile RNN persistent code using NVRTC"};

  if (parameters.algo_ != CUDNN_RNN_ALGO_PERSIST_DYNAMIC)
  {
    return false;
  }

  handle_build_dynamic(
    cudnnBuildRNNDynamic(
      handle.handle_,
      descriptor.descriptor_,
      parameters.batch_size_));

  return handle_build_dynamic.is_success();
}


} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork