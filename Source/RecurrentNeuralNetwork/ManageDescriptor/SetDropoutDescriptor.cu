#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "DropoutDescriptor.h"
#include "SetDropoutDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using DeepNeuralNetwork::CuDNNLibraryHandle;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

SetDropoutDescriptor::SetDropoutDescriptor(
  const float dropout_probability,
  const unsigned long long seed
  ):
  dropout_{dropout_probability},
  seed_{seed}
{}

HandleUnsuccessfulCuDNNCall SetDropoutDescriptor::set_descriptor(
  DropoutDescriptor& descriptor,
  CuDNNLibraryHandle& handle)
{
  HandleUnsuccessfulCuDNNCall handle_set_descriptor {
    "Failed to set dropout descriptor"};

  if (descriptor.is_states_size_known_)
  {
    handle_set_descriptor(cudnnSetDropoutDescriptor(
      descriptor.descriptor_,
      handle.handle_,
      dropout_,
      descriptor.states_,
      descriptor.states_size_,
      seed_));

    return handle_set_descriptor;
  }
  else
  {
    // cuDNN library was not initialized properly; in this case this is when
    // we don't have the states size for dropout.
    handle_set_descriptor(CUDNN_STATUS_NOT_INITIALIZED);

    return handle_set_descriptor;
  }
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork