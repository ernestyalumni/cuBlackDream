#ifndef RECURRENT_NEURAL_NETWORK_PARAMETERS_H
#define RECURRENT_NEURAL_NETWORK_PARAMETERS_H

#include <cstddef>
#include <cstdint>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDescriptor_v8
struct Parameters
{
  Parameters(
    cudnnRNNAlgo_t algorithm,
    cudnnRNNMode_t cell_mode,
    cudnnRNNBiasMode_t bias_mode,
    cudnnDirectionMode_t direction_mode,
    cudnnRNNInputMode_t input_mode,
    cudnnDataType_t data_type,
    cudnnDataType_t math_precision,
    cudnnMathType_t math_operation_type,
    const std::size_t input_size,
    const std::size_t hidden_size,
    const std::size_t projection_size,
    const std::size_t number_of_layers,
    const std::size_t maximum_sequence_length,
    const std::size_t batch_size,
    const cudnnRNNDataLayout_t layout =
      CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
    const uint32_t auxiliary_flags =
      static_cast<uint32_t>(CUDNN_RNN_PADDED_IO_DISABLED));

  inline int get_bidirectional_scale() const
  {
    return (direction_mode_ == CUDNN_BIDIRECTIONAL ? 2 : 1);
  }

  inline int get_input_tensor_size() const
  {
    return maximum_sequence_length_ * batch_size_ * input_size_;
  }

  // Consider size bH if LSTM projection disabled, bP otherwise.
  int get_output_tensor_size() const;

  //------------------------------------------------------------------------
  /// \brief This gets the total size of the hidden tensor that acts on a
  /// single time point, t.
  /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward
  /// hx. Pointer to GPU buffer with RNN initial hidden state. Data
  /// dimensions are described by hDesc tensor descriptor.
  /// hy. Pointer to GPU buffer where final RNN hidden state should be
  /// stored. Data dimensions described by hDesc tensor descriptor.
  //------------------------------------------------------------------------
  int get_hidden_tensor_size() const;

  // LSTM networks only.
  inline int get_cell_tensor_size() const
  {
    return
      number_of_layers_ *
      get_bidirectional_scale() *
      batch_size_ *
      hidden_size_;
  }

  template <typename T>
  inline std::size_t get_total_memory_consumption() const
  {
    return (
      2 * get_input_tensor_size() +
      2 * get_output_tensor_size() +
      8 * get_hidden_tensor_size()) * sizeof(T);
  }

  //----------------------------------------------------------------------------
  /// \return False if we find that LSTM projection is enabled when cell mode
  /// isn't LSTM.
  //----------------------------------------------------------------------------
  bool check_for_valid_parameters();

  //----------------------------------------------------------------------------
  /// 3.1.2.22. cudnnRNNAlgo_t
  /// Input RNN algo (CUDNN_RNN_ALGO_STANDARD, each RNN layer executed as
  /// sequence of operations, algorithm expected to have robust performance
  /// across wide range of network parameters,
  /// CUDNN_RNN_ALGO_PERSIST_STATIC, recurrent parts of network executed using
  /// persistent kernel approach; expected to be fast when first dim. of input
  /// tensor is small (i.e. small minibatch), or
  /// CUDNN_RNN_ALGO_PERSIST_DYNAMIC, expected to be fast when first dim. of
  /// input tensor is small as well, persisent kernels prepared at runtime)
  //----------------------------------------------------------------------------
  cudnnRNNAlgo_t algo_;
  //----------------------------------------------------------------------------
  /// Specifies RNN cell type in the entire model (CUDNN_RNN_RELU,
  /// CUDNN_RNN_TANH, CUDNN_LSTM, CUDNN_GRU)
  //----------------------------------------------------------------------------
  cudnnRNNMode_t cell_mode_;
  //----------------------------------------------------------------------------
  /// Sets number of bias vectors (CUDNN_RNN_NO_BIAS,
  /// CUDNN_RNN_SINGLE_INP_BIAS,
  /// CUDNN_RNN_SINGLE_REC_BIAS, CUDNN_RNN_DOUBLE_BIAS)
  //----------------------------------------------------------------------------
  cudnnRNNBiasMode_t bias_mode_;
  //----------------------------------------------------------------------------
  /// Specifies recurrence pattern;
  /// CUDNN_UNIDIRECTIONAL or CUDNN_BIDIRECTIONAL.
  /// In bidirectional RNNs, hidden states passed between physical layers
  /// are concatenations of forward and backward hidden states.
  //----------------------------------------------------------------------------
  cudnnDirectionMode_t direction_mode_;

  //----------------------------------------------------------------------------
  /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNInputMode_t
  /// Species how input to RNN model is processed by first layer.
  /// CUDNN_LINEAR_INPUT - original input vectors of size inputSize are
  /// multiplied by weight matrix (matrix multiplication) to obtain vectors
  /// of hiddenSize. (input of first recurrent layer)
  /// When inputMode is CUDNN_SKIP_INPUT, original input vectors to first
  /// layer are used as is without multiplying them by weight matrix.
  /// If CUDNN_SKIP_INPUT is used, leading dimension of input tensor must be
  /// equal to hidden state size of network.
  //----------------------------------------------------------------------------
  cudnnRNNInputMode_t input_mode_;

  //----------------------------------------------------------------------------
  /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDataType_t
  /// cudnnDataType_t - enumerated type indicating data type which tensor
  /// descriptor or filter descriptor refers.
  /// CUDNN_DATA_FLOAT - data is 32-bit 
  /// Specifies data type for RNN weights/biases and input and output data.
  //----------------------------------------------------------------------------
  cudnnDataType_t data_type_;

  //----------------------------------------------------------------------------
  /// Used to control compute math precision in RNN model.
  /// For input/output in FP16, mathPrec can be CUDNN_DATA_HALF or
  /// CUDNN_DATA_FLOAT.
  /// For input/output in FP32, parameter can only be CUDNN_DATA_FLOAT.
  //----------------------------------------------------------------------------
  cudnnDataType_t math_precision_;
  //----------------------------------------------------------------------------
  /// Sets preferred option to use NVIDIA Tensor Cores
  /// When dataType, i.e. data_type_, is
  /// CUDNN_DATA_HALF, mathType, i.e. math_type_, can be CUDNN_DEFAULT_MATH
  /// or CUDNN_TENSOR_OP_MATH. ALLOW_CONVERSION setting is treated the same
  /// as CUDNN_TENSOR_OP_MATH for this type.
  /// dataType CUDNN_DATA_FLOAT, mathType can be CUDNN_DEFAULT_MATH or
  /// CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION.
  //----------------------------------------------------------------------------
  cudnnMathType_t math_type_;

  //----------------------------------------------------------------------------
  /// Size of input vector in RNN model.
  /// When inputMode, i.e. input_mode_, =CUDNN_SKIP_INPUT, inputSize, i.e.
  /// input_size_, should match hiddenSize, i.e. hidden_size_.
  //----------------------------------------------------------------------------
  int32_t input_size_;
  //----------------------------------------------------------------------------
  /// hiddenSize - Input. Size of hidden state vector in RNN model. Same hidden
  /// size used in all RNN layers.
  //----------------------------------------------------------------------------
  int32_t hidden_size_;
  //----------------------------------------------------------------------------
  /// projSize - Input. Size of LSTM cell output after recurrent projection.
  /// This value shouln't be larger than hiddenSize. It's legal to set projSize
  /// equal to hiddenSize.
  //----------------------------------------------------------------------------
  int32_t projection_size_;
  //----------------------------------------------------------------------------
  /// numLayers - Input. Number of stacked, physical layers in the deep RNN
  /// model. When dirMode = CUDNN_BIDIRECTIONAL, physical layer consists of
  /// two pseudo-layers corresponding to formward and backward directions.
  //----------------------------------------------------------------------------
  int32_t number_of_layers_;

  //----------------------------------------------------------------------------
  /// \href https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDataDescriptor
  /// \ref 7.2.47. cudnnSetRNNDataDescriptor()
  /// maximum sequence length within RNN data tensor. In unpacked (padded)
  /// layout, this should include padding vectors in each sequence. In
  /// packed (unpadded) layout, this should be equal to greatest element in
  /// seqLengthArray.
  //----------------------------------------------------------------------------
  int maximum_sequence_length_;

  //----------------------------------------------------------------------------
  /// \href https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDataDescriptor
  /// \ref 7.2.47. cudnnSetRNNDataDescriptor()
  /// Number of sequences within a mini-batch.
  //----------------------------------------------------------------------------
  int batch_size_;

  //----------------------------------------------------------------------------
  /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDescriptor_v8
  /// Currently, this parameter is used to enable or disable padded input/output
  /// (CUDNN_RNN_PADDED_IO_DISABLED, CUDNN_RNN_PADDED_IO_ENABLED)
  /// When padded I/O is enabled, layouts\
  /// CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, and
  /// CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED are permitted in RNN data
  /// descriptors.
  //----------------------------------------------------------------------------
  uint32_t auxiliary_flags_;

  cudnnRNNDataLayout_t layout_;
};

struct DefaultParameters : public Parameters
{
  using Parameters::Parameters;

  DefaultParameters();
};

struct LSTMDefaultParameters : public Parameters
{
  using Parameters::Parameters;


  LSTMDefaultParameters() = delete;
  LSTMDefaultParameters(
    const std::size_t input_size,
    const std::size_t hidden_size,
    const std::size_t projection_size,
    const std::size_t number_of_layers,
    const std::size_t maximum_sequence_length,
    const std::size_t batch_size);
};

} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_PARAMETERS_H