#ifndef RECURRENT_NEURAL_NETWORK_MODULES_HIDDEN_H
#define RECURRENT_NEURAL_NETWORK_MODULES_HIDDEN_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstdint>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{
namespace Modules
{

class GetWeightsAndBias
{
  public:

   using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    GetWeightsAndBias(const RecurrentNeuralNetwork::Parameters& parameters);

    //--------------------------------------------------------------------------
    /// \ref 7.2.31. cudnnGetRNNWeightParams()
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
    /// cudnnStatus_t cudnnGetRNNWeightParams(
    ///   cudnnHandle_t handle,
    ///   cudnnRNNDescriptor_t rnnDesc,
    ///   int32_t pseudoLayer,
    ///   size_t weightSpaceSize,
    ///   const void* weightSpace,
    ///   int32_t linlayerID,
    ///   cudnnTensorDescriptor_t mDesc,
    ///   void **mAddr,
    ///   cudnnTensorDescriptor_t bDesc,
    ///   void **bAddr
    ///);
    /// \param [in] handle
    /// \param [in] rnnDesc
    /// \param [in] pseudoLayer - pseudo layer to query.
    /// In unidirectional RNNs, a pseudo-layer is same as a physical layer (
    /// pseudoLAyer=0 is the RNN input layer, pseudoLayer=1 is first hidden
    /// layer). In bidirectional RNNs, there's twice as many pseudo-layers in
    /// comparison to physical layers:
    /// * pseudoLayer=0 refers to the forward direction sub-layer of physical
    /// input layer.
    ///
    /// \param [in] weightSpaceSize
    /// \param [in] weightSpace
    /// \param [in] linLayerID
    /// If cellMode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH,
    /// * 0 references weight matrix or bias vector used in conjunction with
    /// input from previous layer or input to RNN model.
    /// * 1 references weight matrix or bias vector used in conjunction with
    /// hidden state from previous time step or initial hidden state.
    /// \param [out] mDesc - Handle to previously created tensor descriptor.
    /// Shape of corresponding weight matrix is returned in this descriptor in
    /// following format: dimA[3] = {1, rows, cols}. Reported number of tensor
    /// dimensions is zero when weight matrix does not exist.
    /// \param [out] mAddr - Pointer to beginning of weight matrix within weight
    /// space buffer. When weight matrix doesn't exist, returned address is
    /// NULL.
    /// \param [out] bDesc - Handle to previously created tensor descriptor.
    /// Shape of corresponding bias vector returned in this descriptor in
    /// following format: dimA[3] = {1, rows, 1}.
    /// \param [out] bAddr - Pointer to beginning of bias vector within weight
    /// space buffer. When bias vector doesn't exist, returned address is NULL.
    ///
    /// \param [out] m_tensor_descriptor - Shape of the weight matrix is
    /// returned in this descriptor. Reported number of tensor dimensions is 0
    /// when weight matrix doesn't exist.
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCuDNNCall get_weight_and_bias(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& descriptor,
      const int32_t pseudo_layer,
      RecurrentNeuralNetwork::WeightSpace& weight_space,
      const int32_t linear_layer_ID,
      Tensors::ManageDescriptor::TensorDescriptor& m_tensor_descriptor,
      Tensors::ManageDescriptor::TensorDescriptor& b_tensor_descriptor);

    HandleUnsuccessfulCuDNNCall get_weight_and_bias(
      RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
        descriptors,
      const int32_t pseudo_layer,
      RecurrentNeuralNetwork::WeightSpace& weight_space,
      const int32_t linear_layer_ID,
      Tensors::ManageDescriptor::TensorDescriptor& m_tensor_descriptor,
      Tensors::ManageDescriptor::TensorDescriptor& b_tensor_descriptor);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
    /// \ref 7.2.31. cudnnGetRNNWeightParams()
    /// While the API says the types for mAddr, bAddr, pointer to the beginning
    /// of weight matrix, bias vector, within weight space buffer, respectively,
    /// what you actually want is a pointer of type void* and to pass its
    /// address in.
    //--------------------------------------------------------------------------
    void* weight_matrix_address_;
    void* bias_address_;

  private:

    int bidirectional_scale_;
    int32_t hidden_size_;
    cudnnRNNMode_t cell_mode_;
};

} // namespace Modules
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MODULES_INPUT_H