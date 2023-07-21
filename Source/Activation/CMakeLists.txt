SET(ACTIVATION_CU_SOURCES
  ManageDescriptor/ActivationDescriptor.cu
  ManageDescriptor/SetDescriptor.cu
  )

ADD_LIBRARY(ActivationCU
  ${ACTIVATION_CU_SOURCES}
  )

# https://forums.developer.nvidia.com/t/cudnn-linking-error-using-cmake/57968/3

TARGET_LINK_LIBRARIES(ActivationCU
  cudnn
  )
