SET(TRANSFORMER_CU_SOURCES
  Attention/Parameters.cu
  Attention/WeightWorkAndReserveSpaces.cu
  ManageDescriptor/AttentionDescriptor.cu
  ManageDescriptor/LibraryHandleDropoutsAttention.cu
  ManageDescriptor/SequenceDataDescriptor.cu
  ManageDescriptor/SetAttentionDescriptor.cu
  ManageDescriptor/SetSequenceDataDescriptor.cu
  )

ADD_LIBRARY(TransformerCU
  ${TRANSFORMER_CU_SOURCES}
  )

# https://forums.developer.nvidia.com/t/cudnn-linking-error-using-cmake/57968/3

TARGET_LINK_LIBRARIES(TransformerCU
  cudnn
  )
