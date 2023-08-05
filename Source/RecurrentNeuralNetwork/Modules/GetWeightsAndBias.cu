#include "GetWeightsAndBias.h"

#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"

#include <cudnn.h>

using DeepNeuralNetwork::CuDNNLibraryHandle;
using RecurrentNeuralNetwork::ManageDescriptor::Descriptor;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::Parameters;
using RecurrentNeuralNetwork::WeightSpace;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;
using Tensors::ManageDescriptor::TensorDescriptor;

namespace RecurrentNeuralNetwork
{
namespace Modules
{

GetWeightsAndBias::GetWeightsAndBias(
  const Parameters& parameters
  ):
  weight_matrix_address_{nullptr},
  bias_address_{nullptr},
  bidirectional_scale_{parameters.get_bidirectional_scale()},
  hidden_size_{parameters.hidden_size_},
  cell_mode_{parameters.cell_mode_}
{}

HandleUnsuccessfulCuDNNCall GetWeightsAndBias::get_weight_and_bias(
  CuDNNLibraryHandle& handle,
  Descriptor& descriptor,
  const int32_t pseudo_layer,
  WeightSpace& weight_space,
  const int32_t linear_layer_ID,
  TensorDescriptor& m_tensor_descriptor,
  TensorDescriptor& b_tensor_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_get_weight {
    "Failed to get weight parameters"};

  //----------------------------------------------------------------------------
  /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
  /// CUDNN_STATUS_BAD_PARAM if invalid input argument; e.g. value of
  /// pseudoLayer out of range or linLayerID is negative or larger than 8.
  //----------------------------------------------------------------------------
  if (linear_layer_ID < 0 || linear_layer_ID > 8)
  {
    handle_get_weight(CUDNN_STATUS_BAD_PARAM);
    return handle_get_weight;  
  }
  
  handle_get_weight(cudnnGetRNNWeightParams(
    handle.handle_,
    descriptor.descriptor_,
    pseudo_layer,
    weight_space.get_weight_space_size(),
    weight_space.weight_space_,
    linear_layer_ID,
    m_tensor_descriptor.descriptor_,
    &weight_matrix_address_,
    b_tensor_descriptor.descriptor_,
    &bias_address_));

  // It was empirically found that if linLayerID i.e. linear_layer_ID was out of
  // bounds, it would not yield CUDNN_STATUS_BAD_PARAM but rather return
  // nullptrs.
  if ((weight_matrix_address_ == nullptr) ||
    (bias_address_ == nullptr))
  {
    handle_get_weight(CUDNN_STATUS_BAD_PARAM);
  }

  return handle_get_weight;
}

HandleUnsuccessfulCuDNNCall GetWeightsAndBias::get_weight_and_bias(
  LibraryHandleDropoutRNN& descriptors,
  const int32_t pseudo_layer,
  WeightSpace& weight_space,
  const int32_t linear_layer_ID,
  TensorDescriptor& m_tensor_descriptor,
  TensorDescriptor& b_tensor_descriptor)
{
  return get_weight_and_bias(
    descriptors.handle_,
    descriptors.descriptor_,
    pseudo_layer,
    weight_space,
    linear_layer_ID,
    m_tensor_descriptor,
    b_tensor_descriptor);
}

} // namespace Modules
} // namespace RecurrentNeuralNetwork