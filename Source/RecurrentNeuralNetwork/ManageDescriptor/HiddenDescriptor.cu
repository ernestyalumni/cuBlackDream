#include "HiddenDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"

#include <cudnn.h>
#include <stdexcept>

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

HiddenDescriptor3Dim::HiddenDescriptor3Dim(
  const RecurrentNeuralNetwork::Parameters& parameters
  ):
  descriptor_{},
  set_for_3D_tensor_{}
{
  set_for_3D_tensor_.set_for_hidden_layers(parameters);
  const auto result =
    set_for_3D_tensor_.set_descriptor(descriptor_, parameters);
  if (!result.is_success())
  {
    throw std::runtime_error(result.get_error_message());
  }
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
