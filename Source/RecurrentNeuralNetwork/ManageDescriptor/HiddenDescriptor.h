#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_HIDDEN_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_HIDDEN_DESCRIPTOR_H

#include "RecurrentNeuralNetwork/Parameters.h"
#include "Tensors/ManageDescriptor/SetForNDTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"

#include <cstddef>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
/// N - dimension of the tensor.
//------------------------------------------------------------------------------
template <std::size_t N>
struct HiddenDescriptor
{
  using HandleUnsuccessfulCuDNNCall =
    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

  HiddenDescriptor():
    descriptor_{},
    set_for_ND_tensor_{}
  {}

  ~HiddenDescriptor() = default;

  HandleUnsuccessfulCuDNNCall set_descriptor(
    const RecurrentNeuralNetwork::Parameters& parameters)
  {
    return set_for_ND_tensor_.set_descriptor(descriptor_, parameters);
  }

  Tensors::ManageDescriptor::TensorDescriptor descriptor_;
  Tensors::ManageDescriptor::SetForNDTensor<N> set_for_ND_tensor_;
};

struct HiddenDescriptor3Dim
{
  HiddenDescriptor3Dim();

  ~HiddenDescriptor() = default;

  Tensors::ManageDescriptor::TensorDescriptor descriptor_;
  Tensors::ManageDescriptor::SetFor3DTensor set_for_3D_tensor_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_HIDDEN_DESCRIPTOR_H