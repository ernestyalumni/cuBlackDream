SET(TENSORS_CU_SOURCES
  ManageDescriptor/SetFor4DTensor.cu
  ManageDescriptor/SetForNDTensor.cu
  ManageDescriptor/TensorDescriptor.cu
  )

ADD_LIBRARY(TensorsCU
  ${TENSORS_CU_SOURCES}
  )

# https://forums.developer.nvidia.com/t/cudnn-linking-error-using-cmake/57968/3

TARGET_LINK_LIBRARIES(TensorsCU
  cudnn
  )
