#ifndef RECURRENT_NEURAL_NETWORK_MODULES_WEIGHT_H
#define RECURRENT_NEURAL_NETWORK_MODULES_WEIGHT_H

#include "Algebra/Modules/Vectors/RunTimeVoidPtrArray.h"
#include "RecurrentNeuralNetwork/Modules/GetWeightsAndBias.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "Utilities/CuDNNDataTypeToType.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cstddef>
#include <stdexcept>

namespace RecurrentNeuralNetwork
{
namespace Modules
{

class Weight
{
  public:

    using HandleUnsuccessfulCUDACall =
      Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

    Weight(Tensors::ManageDescriptor::TensorDescriptor& tensor_descriptor);

    ~Weight() = default;

    std::size_t get_total_size() const;

    HandleUnsuccessfulCUDACall copy_from(
      GetWeightsAndBias& get_weights_and_bias);

    Algebra::Modules::Vectors::RunTimeVoidPtrArray w_;

    std::size_t number_of_rows_;
    std::size_t number_of_columns_;
    cudnnDataType_t data_type_;
};

} // namespace Modules
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MODULES_WEIGHT_H