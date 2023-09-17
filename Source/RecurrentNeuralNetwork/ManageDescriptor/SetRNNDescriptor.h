#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_RNN_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_RNN_DESCRIPTOR_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "Descriptor.h"
#include "DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

//--------------------------------------------------------------------------
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDescriptor_v8
/// \ref 7.2.49. cudnnSetRNNDescriptor_v8()
/// cudnnStatus_t cudnnSetRNNDescriptor_v8(
///   cudnnRNNDescriptor_t rnnDesc,
///   cudnnRNNAlgo_t algo,
///   cudnnRNNMode_t cellMode,
///   cudnnRNNBiasMode_t biasMode,
///   cudnnDirectionMode_t dirMode,
///   cudnnRNNInputMode_t inputMode,
///   cudnnDataType_t dataType,
///   cudnnMathType_t mathType,
///   int32_t inputSize,
///   int32_t hiddenSize,
///   int32_t projSize,
///   int32_t numLayers,
///   cudnnDropoutDescriptor_t dropoutDesc,
///   uint32_t auxFlags);
/// auxFlags - Used to pass miscellaneous switches that don't require
/// additional numerical values to configure corresponding feature.
/// Currently, this parameter is used to enable or disable padded input/
/// output:
/// CUDNN_RNN_PADDED_IO_DISABLED, CUDNN_RNN_PADDED_IO_ENABLED.
/// When padded I/O is enaabled, layouts
/// CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED and
/// CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED are permitted in RNN data
/// descriptors.
/// \param [in] numLayers - number of stacked physical layers in the deep
/// RNN model.
//--------------------------------------------------------------------------

Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall set_rnn_descriptor(
  Descriptor& descriptor,
  const RecurrentNeuralNetwork::Parameters& parameters,
  DropoutDescriptor& dropout_descriptor);

//--------------------------------------------------------------------------
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBuildRNNDynamic
/// \ref 7.2.2. cudnnBuildRNNDynamic()
/// cudnnBuildRNNDynamic() compiles RNN persistent code using CUDA runtime
/// compilation library (NVRTC) when CUDNN_RNN_ALGO_PERSIST_DYNAMIC algo is
/// selected. Code is tailored to current GPU and specific hyperparameters
/// (miniBatch). This call is expected to be expensive in terms of runtime
/// and should be invoked infrequently.
/// \return True if RNN persistent code compiled using CUDA runtime
/// compilation library (NVRTC) successfully; return False if it's not
/// dynamic or if it's unsuccessful.
//--------------------------------------------------------------------------
bool build_with_NVRTC_if_dynamic(
    DeepNeuralNetwork::CuDNNLibraryHandle& handle,
    Descriptor& descriptor,
    const RecurrentNeuralNetwork::Parameters& parameters);

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_RNN_DESCRIPTOR_H