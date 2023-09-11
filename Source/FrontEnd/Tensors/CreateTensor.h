#ifndef FRONT_END_TENSORS_CREATE_TENSOR_H
#define FRONT_END_TENSORS_CREATE_TENSOR_H

#include <array>
#include <cstddef>
#include <cudnn.h>
#include <cudnn_frontend.h>

namespace FrontEnd
{
namespace Tensors
{

//------------------------------------------------------------------------------
/// https://nvidia.github.io/cudnn-frontend/d3/d3c/group__TensorBuilder__v8.html
/// https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/mha_sample.cpp
/// \details Rank is the "number of dimensions" of the tensor itself.
/// \param[in] is_virtual Whether it is an intermediate tensor of an op
/// (operation) graph. See
/// https://github.com/NVIDIA/cudnn-frontend/blob/main/include/cudnn_frontend_Tensor.h#L169
/// \param[in] is_value Whether tensor is in host memory that needs to be passed
/// to the kernel by value. See
/// https://github.com/NVIDIA/cudnn-frontend/blob/main/include/cudnn_frontend_Tensor.h#L170
//------------------------------------------------------------------------------

template <std::size_t Rank>
cudnn_frontend::Tensor create_tensor(
  const cudnnDataType_t data_type,
  const int64_t id,
  const std::array<int64_t, Rank>& dimensions,
  const std::array<int64_t, Rank>& strides,
  const int64_t alignment,
  const bool is_virtual,
  const bool is_value = false
  )
{
  return cudnn_frontend::TensorBuilder()
    .setDim(dimensions.size(), dimensions.data())
    .setStride(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(data_type)
    .setVirtual(is_virtual)
    .setByValue(is_value)
    .build();
}

template <std::size_t Rank>
cudnn_frontend::Tensor create_tensor(
  const cudnnDataType_t data_type,
  const int64_t id,
  const std::array<int64_t, Rank>& dimensions,
  const std::array<int64_t, Rank>& strides,
  const int64_t alignment
  )
{
  return cudnn_frontend::TensorBuilder()
    .setDim(dimensions.size(), dimensions.data())
    .setStride(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(data_type)
    .build();
}

template <std::size_t Rank>
cudnn_frontend::Tensor create_tensor(
  const cudnnDataType_t data_type,
  const int64_t id,
  const std::array<int64_t, Rank>& dimensions,
  const std::array<int64_t, Rank>& strides,
  const int64_t alignment,
  const cudnn_frontend::cudnnBackendTensorReordering_t reorder_type,
  const bool is_virtual,
  const bool is_value = false
  )
{
  return cudnn_frontend::TensorBuilder()
    .setDim(dimensions.size(), dimensions.data())
    .setStride(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(data_type)
    .setVirtual(is_virtual)
    .setByValue(is_value)
    .setReorderType(reorder_type)
    .build();
}

} // namespace Tensors
} // namespace FrontEnd

#endif // FRONT_END_TENSORS_CREATE_TENSOR_H