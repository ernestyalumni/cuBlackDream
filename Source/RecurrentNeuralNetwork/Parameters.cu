#include "Parameters.h"

#include <cstddef>
#include <cudnn.h>
#include <stdexcept>

using std::runtime_error;
using std::size_t;

namespace RecurrentNeuralNetwork
{

Parameters::Parameters(
  cudnnRNNAlgo_t algorithm,
  cudnnRNNMode_t cell_mode,
  cudnnRNNBiasMode_t bias_mode,
  cudnnDirectionMode_t direction_mode,
  cudnnRNNInputMode_t input_mode,
  cudnnDataType_t data_type,
  cudnnDataType_t math_precision,
  cudnnMathType_t math_operation_type,
  const size_t input_size,
  const size_t hidden_size,
  const size_t projection_size,
  const size_t number_of_layers,
  const size_t maximum_sequence_length,
  const size_t batch_size,
  const uint32_t auxiliary_flags
  ):
  algo_{algorithm},
  cell_mode_{cell_mode},
  bias_mode_{bias_mode},
  direction_mode_{direction_mode},
  input_mode_{input_mode},
  data_type_{data_type},
  math_precision_{math_precision},
  math_type_{math_operation_type},
  input_size_{static_cast<int32_t>(input_size)},
  hidden_size_{static_cast<int32_t>(hidden_size)},
  projection_size_{static_cast<int32_t>(projection_size)},
  number_of_layers_{static_cast<int32_t>(number_of_layers)},
  maximum_sequence_length_{static_cast<int>(maximum_sequence_length)},
  batch_size_{static_cast<int>(batch_size)},
  auxiliary_flags_{auxiliary_flags}
{
  if (data_type_ == CUDNN_DATA_HALF)
  {
    if ((math_type_ != CUDNN_DEFAULT_MATH) &&
      (math_type_ != CUDNN_TENSOR_OP_MATH))
    {
      throw runtime_error(
        "Since data type is CUDNN_DATA_HALF, math type is default or tensor op only."
      );
    }
  }

  if (input_mode_ == CUDNN_SKIP_INPUT)
  {
    if (input_size_ != hidden_size_)
    {
      throw runtime_error(
        "Since input mode is CUDNN_SKIP_INPUT, input_size = hidden_size");
    }
  }

  if (projection_size > hidden_size)
  {
    throw runtime_error(
      "[ERROR] Inconsistent parameter: projSize is larger than hiddenSize!");
  }
}

DefaultParameters::DefaultParameters():
  Parameters{
    CUDNN_RNN_ALGO_STANDARD,
    CUDNN_RNN_RELU,
    CUDNN_RNN_DOUBLE_BIAS,
    CUDNN_UNIDIRECTIONAL,
    CUDNN_LINEAR_INPUT,
    CUDNN_DATA_FLOAT,
    CUDNN_DATA_FLOAT,
    CUDNN_DEFAULT_MATH,
    512,
    512,
    512,
    2,
    20,
    64,
    CUDNN_RNN_PADDED_IO_DISABLED}
{}

} // namespace RecurrentNeuralNetwork
