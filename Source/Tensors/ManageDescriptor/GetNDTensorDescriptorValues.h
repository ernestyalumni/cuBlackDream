#ifndef TENSORS_MANAGE_DESCRIPTOR_GET_ND_TENSOR_DESCRIPTOR_VALUES_H
#define TENSORS_MANAGE_DESCRIPTOR_GET_ND_TENSOR_DESCRIPTOR_VALUES_H

#include "TensorDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <algorithm>
#include <cstddef>
#include <cudnn.h>
#include <type_traits>

namespace Tensors
{
namespace ManageDescriptor
{

template <std::size_t N, typename = std::enable_if_t<(N > 0)>>
class GetNDTensorDescriptorValues
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    //--------------------------------------------------------------------------
    /// \details EY 20230805
    //--------------------------------------------------------------------------
    GetNDTensorDescriptorValues():
      data_type_{new cudnnDataType_t},
      nb_dims_{new int[N]},
      dimensions_array_{new int[N]},
      strides_array_{new int[N]}
    {
      std::fill(nb_dims_, nb_dims_ + N, -1);
      std::fill(dimensions_array_, dimensions_array_ + N, -1);
      std::fill(strides_array_, strides_array_ + N, -1);
    }

    ~GetNDTensorDescriptorValues()
    {
      delete data_type_;
      delete [] nb_dims_;
      delete [] dimensions_array_;
      delete [] strides_array_;
    }

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
    /// \ref 3.2.61. cudnnGetTensorNdDescriptor()
    /// \brief Retrieves values stored in descriptor object.
    /// cudnnStatus_t cudnnGetTensorNdDescriptor(
    ///   const cudnnTensorDescriptor_t tensorDesc,
    ///   int nbDimsRequested,
    ///   cudnnDataType_t* dataType,
    ///   int* nbDims,
    ///   int dimA[],
    /// )
    /// \param [in] tensorDesc - Handle to a previously initialized tensor
    // descriptor.
    /// \param [out] nbDims - Actual number of dimensions of tensor will be
    /// returned in nbDims[0]
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCuDNNCall get_values(
      TensorDescriptor& descriptor,
      const std::size_t number_of_requested_dimensions)
    {
      HandleUnsuccessfulCuDNNCall handle_get_values {
        "Failed to return values stored in tensor descriptor"};

      handle_get_values(cudnnGetTensorNdDescriptor(
        descriptor.descriptor_,
        number_of_requested_dimensions,
        data_type_,
        nb_dims_,
        dimensions_array_,
        strides_array_));

      return handle_get_values;
    }

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDataType_t
    /// \ref 3.1.2.6. cudnnDataType_t
    /// CUDNN_DATA_FLOAT - 32-bit single-precision floating point
    /// CUDNN_DATA_DOUBLE
    /// CUDNN_DATA_HALF
    /// CUDNN_DATA_INT8
    //--------------------------------------------------------------------------
    // Data type.
    cudnnDataType_t* data_type_;
    // Actual number of dimensions of tensor that'll be returned.
    int* nb_dims_;
    // Array of dimensions filled with dimensions from provided tensor
    // descriptor.
    int* dimensions_array_;
    // Array of dimensions filled with strides from provided tensor
    // descriptor.
    int* strides_array_;
};

} // namespace ManageDescriptor
} // namespace Tensors

#endif // TENSORS_MANAGE_DESCRIPTOR_GET_ND_TENSOR_DESCRIPTOR_VALUES_H
