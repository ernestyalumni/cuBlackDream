SET(ALGEBRA_CU_SOURCES
  Modules/Vectors/RunTimeVoidPtrArray.cu
  )

ADD_LIBRARY(AlgebraCU
  ${ALGEBRA_CU_SOURCES}
  )

# https://forums.developer.nvidia.com/t/cudnn-linking-error-using-cmake/57968/3

TARGET_LINK_LIBRARIES(AlgebraCU
  cudnn
  )
