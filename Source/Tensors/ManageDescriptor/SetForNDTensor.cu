#include "SetForNDTensor.h"
#include "RecurrentNeuralNetwork/Parameters.h"

#include <cstddef>
#include <cudnn.h>

namespace Tensors
{
namespace ManageDescriptor
{

SetFor3DTensor::SetFor3DTensor():
  SetForNDTensor<3>{}
{}

SetFor3DTensor::SetFor3DTensor(
    const RecurrentNeuralNetwork::Parameters& parameters):
  SetForNDTensor<3>{}
{
  set_for_hidden_layers(parameters);
}

void SetFor3DTensor::set_for_hidden_layers(
  const RecurrentNeuralNetwork::Parameters& parameters)
{
  dimensions_array_[0] =
    parameters.number_of_layers_ * parameters.get_bidirectional_scale();

  dimensions_array_[1] = parameters.batch_size_;
  dimensions_array_[2] = parameters.hidden_size_;

  strides_array_[0] = dimensions_array_[1] * dimensions_array_[2];
  strides_array_[1] = dimensions_array_[2];
  strides_array_[2] = 1;
}

} // namespace ManageDescriptor
} // namespace Tensors