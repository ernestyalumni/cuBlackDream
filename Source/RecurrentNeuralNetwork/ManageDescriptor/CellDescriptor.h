#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_CELL_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_CELL_DESCRIPTOR_H

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
/// N - rank of the tensor.
//------------------------------------------------------------------------------
template <std::size_t N>
//------------------------------------------------------------------------------
/// \details CellDescriptor is only for LSTM.
//------------------------------------------------------------------------------
struct CellDescriptor
{
  using HandleUnsuccessfulCuDNNCall =
    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

  CellDescriptor() = delete;

  CellDescriptor(const RecurrentNeuralNetwork::Parameters& parameters):
    descriptor_{},
    set_for_ND_tensor_{}
  {
    set_cell_layers_dimensions_for_forward(parameters);
    set_strides_by_dimensions();
  }  

  ~CellDescriptor() = default;

  void set_cell_layers_dimensions_for_forward(
    const RecurrentNeuralNetwork::Parameters& parameters)
  {
    set_for_ND_tensor_.set_dimensions_array_value(
      0,
      parameters.number_of_layers_ * parameters.get_bidirectional_scale());

    set_for_ND_tensor_.set_dimensions_array_value(
      1,
      parameters.batch_size_);

    set_for_ND_tensor_.set_dimensions_array_value(
      2,
      parameters.hidden_size_);
  }

  // TODO: Make this more robust, e.g. what if a dimension element was set to 0?
  void set_strides_by_dimensions()
  {
    set_for_ND_tensor_.set_strides_from_dimensions_as_descending();
  }

  HandleUnsuccessfulCuDNNCall set_descriptor(
    const RecurrentNeuralNetwork::Parameters& parameters)
  {
    return set_for_ND_tensor_.set_descriptor(descriptor_, parameters);
  }

  Tensors::ManageDescriptor::TensorDescriptor descriptor_;
  Tensors::ManageDescriptor::SetForNDTensor<N> set_for_ND_tensor_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_CELL_DESCRIPTOR_H