#ifndef TENSORS_MANAGE_DESCRIPTOR_SET_FOR_ND_TENSOR_H
#define TENSORS_MANAGE_DESCRIPTOR_SET_FOR_ND_TENSOR_H

#include "RecurrentNeuralNetwork/Parameters.h"
#include "TensorDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cudnn.h>
#include <type_traits>

namespace Tensors
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
/// N - dimension of the tensor.
//------------------------------------------------------------------------------
template <std::size_t N, typename = std::enable_if_t<(N > 0)>>
class SetForNDTensor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    SetForNDTensor():
      dimensions_array_{},
      strides_array_{}
    {}

    ~SetForNDTensor() = default;

    //-------------------------------------------------------------------------- 
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor
    /// \ref 3.2.95. cudnnSetTensorNdDescriptor()
    /// cudnnStatus_t cudnnSetTensorNdDescriptor(
    ///   cudnnTensorDescriptor_t tensorDesc,
    ///   cudnnDataType_t dataType,
    ///   int nbDims,
    ///   const int dimA[],
    ///   const int strideA[])
    /// nbDims - Dimension of the tensor.
    /// Do not use 2 dimensions. Due to historical reasons, the minimum number
    /// of dimensions in the filter is 3. For more info, refer to
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNLinLayerBiasParams
    /// cudnnGetRNNLinLayerBiasParams().
    /// dimA - Array of dimension nbDims that contain the size of the tensor for
    /// every dimension. The size along unused dimensions should be set to 1. By
    /// convention, the ordering of dimensions in the array follows the format -
    /// [N, C, D, H, W], with W occupying the smallest index in the array.
    /// strideA - Array of dimension nbDims that contain stride of tensor for
    /// every dimension. By convention, ordering of the strides in the array
    /// follows the format -
    /// [Nstride, Cstride, Dstride, Hstride, Wstride], with Wstride occupying
    /// smallest index in array.
    //-------------------------------------------------------------------------- 

    HandleUnsuccessfulCuDNNCall set_descriptor(
      TensorDescriptor& tensor_descriptor,
      const RecurrentNeuralNetwork::Parameters& parameters)
    {
      HandleUnsuccessfulCuDNNCall handle_set_descriptor {
        "Failed to set Tensor ND descriptor"};

      handle_set_descriptor(cudnnSetTensorNdDescriptor(
        tensor_descriptor.descriptor_,
        parameters.data_type_,
        N,
        dimensions_array_,
        strides_array_));

      return handle_set_descriptor;
    }

    friend class HiddenDescriptor;
    friend class SetFor3DTensor;
    friend class SetFor3DCellTensor;

    inline int get_dimensions_array_value(const std::size_t i) const
    {
      return dimensions_array_[i];
    }

    inline void set_dimensions_array_value(const std::size_t i, const int value)
    {
      dimensions_array_[i] = value;
    }

    inline int get_strides_array_value(const std::size_t i) const
    {
      return strides_array_[i];
    }

    inline void set_strides_array_value(const std::size_t i, const int value)
    {
      strides_array_[i] = value;
    }

    void set_dimensions(const std::array<int, N>& input_array)
    {
      std::copy(input_array.begin(), input_array.end(), dimensions_array_);
    }

    void set_strides(const std::array<int, N>& input_array)
    {
      std::copy(input_array.begin(), input_array.end(), strides_array_);      
    }

    void set_strides_from_dimensions_as_descending()
    {
      int product_of_dimensions {1};
      for (std::size_t i {N}; i > 0; --i)
      {
        strides_array_[i - 1] = product_of_dimensions;
        product_of_dimensions *= dimensions_array_[i - 1];
      }
    }

    void set_strides_from_dimensions_as_increasing()
    {
      int product_of_dimensions {1};
      for (std::size_t i {0}; i < N; ++i)
      {
        strides_array_[i] = product_of_dimensions;
        if (i < N - 1)
        {
          product_of_dimensions *= dimensions_array_[i + 1];
        }
      }
    }

  private:

    int dimensions_array_[N];
    int strides_array_[N];
};

class SetFor3DTensor : public SetForNDTensor<3>
{
  public:

    SetFor3DTensor();

    SetFor3DTensor(const RecurrentNeuralNetwork::Parameters& parameters);

    void set_for_hidden_layers(
      const RecurrentNeuralNetwork::Parameters& parameters);
};

class SetFor3DCellTensor : public SetForNDTensor<3>
{
  public:

    SetFor3DCellTensor();

    SetFor3DCellTensor(const RecurrentNeuralNetwork::Parameters& parameters);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward
    /// cDesc Input. For LSTM networks only.
    /// First dimension of tensor depends on dirMode argument passed to
    /// cudnnSetRNNDescriptor_v8() call.
    /// If dirMode is CUDNN_UNIDIRECTIONAL, 1st. dim. should match L, numLayers.
    /// If dirMode is CUDNN_BIDIRECTIONAL, 1st. dim. should match 2*L.
    /// 2nd. tensor dim. must match batchSize in xDesc, N.
    /// 3rd. dim. must match hiddenSize argument, H, passed to
    /// cudnnSetRNNDescriptor_v8() call.
    //--------------------------------------------------------------------------
    void set_cell_layers_dimensions_for_forward(
      const RecurrentNeuralNetwork::Parameters& parameters);
};

} // namespace ManageDescriptor
} // namespace Tensors

#endif // TENSORS_MANAGE_DESCRIPTOR_SET_FOR_ND_TENSOR_H