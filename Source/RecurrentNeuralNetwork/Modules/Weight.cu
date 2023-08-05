#include "Weight.h"

#include "RecurrentNeuralNetwork/Modules/GetWeightsAndBias.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "Tensors/ManageDescriptor/GetNDTensorDescriptorValues.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"
#include "Utilities/CuDNNDataTypeToSize.h"

#include <cstddef>
#include <cuda_runtime.h> // cudaFree, cudaMalloc
#include <stdexcept>

using RecurrentNeuralNetwork::Modules::GetWeightsAndBias;
using Tensors::ManageDescriptor::GetNDTensorDescriptorValues;
using Tensors::ManageDescriptor::TensorDescriptor;
using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;
using Utilities::cuDNN_data_type_to_size;
using std::runtime_error;

namespace RecurrentNeuralNetwork
{
namespace Modules
{

Weight::Weight(TensorDescriptor& tensor_descriptor):
  w_{},
  number_of_rows_{0},
  number_of_columns_{0},
  data_type_{CUDNN_DATA_FLOAT}
{
  GetNDTensorDescriptorValues<3> get_values {};
  auto result = get_values.get_values(tensor_descriptor, 3);

  if (!result.is_success())
  {
    throw runtime_error(
      "Unable to get values from input tensor descriptor on Weight construction"
      );
  }

  if (get_values.nb_dims_[0] == 0)
  {
    throw runtime_error(
      "No dimensions obtained from input tensor descriptor on Weight construction"
      );
  }

  number_of_rows_ = get_values.dimensions_array_[1];
  number_of_columns_ = get_values.dimensions_array_[2];
  data_type_ = *(get_values.data_type_);

  auto initialize_result = w_.initialize(get_total_size());

  if (!initialize_result.is_cuda_success())
  {
    throw runtime_error(initialize_result.get_error_message());
  }
}

std::size_t Weight::get_total_size() const
{
  const std::size_t data_type_size {cuDNN_data_type_to_size(data_type_)};
  return number_of_rows_ * number_of_columns_ * data_type_size;
}

HandleUnsuccessfulCUDACall Weight::copy_from(
  GetWeightsAndBias& get_weights_and_bias)
{
  HandleUnsuccessfulCUDACall handle_copy {
    "Failed to copy from device weight data"};

  if (get_weights_and_bias.weight_matrix_address_ == nullptr)
  {
    handle_copy(cudaErrorInvalidValue);
    return handle_copy;
  }

  handle_copy(cudaMemcpy(
    w_.values_,
    get_weights_and_bias.weight_matrix_address_,
    get_total_size(),
    cudaMemcpyDeviceToDevice));

  return handle_copy;
}

} // namespace Modules
} // namespace RecurrentNeuralNetwork
