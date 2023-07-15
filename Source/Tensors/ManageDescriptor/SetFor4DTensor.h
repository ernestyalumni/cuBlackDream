#ifndef TENSORS_MANAGE_DESCRIPTOR_SET_FOR_4D_TENSOR_H
#define TENSORS_MANAGE_DESCRIPTOR_SET_FOR_4D_TENSOR_H

#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"
#include "TensorDescriptor.h"

#include <cstddef>
#include <cudnn.h>

namespace Tensors
{
namespace ManageDescriptor
{

class SetFor4DTensor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    //-------------------------------------------------------------------------- 
    /// \brief Wrapper for initializing a previously created generic tensor
    /// descriptor into a 4D tensor.
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor4dDescriptor
    /// \ref 3.2.93. cudnnSetTensor4dDescriptor().
    /// https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-API.pdf
    /// 3.1.2.6 cudnnDataType_t
    /// Values
    /// CUDNN_DATA_FLOAT - 32-bit float
    /// CUDNN_DATA_DOUBLE - 64-bit double
    /// CUDNN_DATA_HALF - 16-bit floating-point
    /// CUDNN_DATA_INT8 - 8-bit signed integer.
    /// CUDNN_DATA_INT32 - 32-bit signed integer.
    ///
    /// 3.1.2.28 cudnnTensorFormat_t
    /// cudnnTensorFormat_t for pre-defined layout.
    /// CUDNN_TENSOR_NCHW - specifies data laid out in following order: batch
    /// size, feature maps, rows, columns. No padding. Columns are inner
    /// dimension and images are outermost dimension.
    /// CUDNN_TENSOR_NHWC - specifies data laid out in following order: batch
    /// size, rows, columns, feature maps.
    /// CUDNN_TENSOR_NCHW_VECT_C - batch size, feature maps, rows, columns.
    /// However, each element of tensor is vector of multiple feature maps.
    /// NCHW INT8x32 format is really N x (C/32) x H x W x 32
    //-------------------------------------------------------------------------- 
    SetFor4DTensor(
      const std::size_t n,
      const std::size_t c,
      const std::size_t h,
      const std::size_t w,
      const cudnnDataType_t data_type,
      const cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);

    ~SetFor4DTensor() = default;

    HandleUnsuccessfulCuDNNCall set_descriptor(
      cudnnTensorDescriptor_t& tensor_descriptor);

    //-------------------------------------------------------------------------- 
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensor4dDescriptor
    //-------------------------------------------------------------------------- 

    // Number of images.
    const int n_;

    // Number of feature maps per image.
    const int c_;

    // Height of each feature map.
    const int h_;

    // Width of each feature map.
    const int w_;

    const int number_of_elements_;

  private:

    cudnnDataType_t data_type_;
    cudnnTensorFormat_t format_;
};

} // namespace ManageDescriptor
} // namespace Tensors

#endif // TENSORS_MANAGE_DESCRIPTOR_SET_FOR_4D_TENSOR_H