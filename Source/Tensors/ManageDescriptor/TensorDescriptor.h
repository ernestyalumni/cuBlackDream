#ifndef TENSORS_MANAGE_DESCRIPTOR_TENSOR_DESCRIPTOR_H
#define TENSORS_MANAGE_DESCRIPTOR_TENSOR_DESCRIPTOR_H

#include <cudnn.h>

namespace Tensors
{

namespace ManageDescriptor
{

class TensorDescriptor
{
  public:

    TensorDescriptor();

    ~TensorDescriptor();

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorDescriptor_t
    /// \ref 3.1.1.11 cudnnTensorDescriptor_t
    /// \details cudnnTensorDescriptor_t is a pointer to an opaque structure
    /// holding the description of a generic n-D dataset.
    //--------------------------------------------------------------------------
    cudnnTensorDescriptor_t descriptor_;
};

} // namespace ManageDescriptor
} // namespace Tensors

#endif // TENSORS_MANAGE_DESCRIPTOR_TENSOR_DESCRIPTOR_H