SET(DEEPNEURALNETWORK_CU_SOURCES
  CuDNNLibraryHandle.cu
  GetCuDNNVersion.cu)

ADD_LIBRARY(DeepNeuralNetworkCU
  ${DEEPNEURALNETWORK_CU_SOURCES}
  )

# https://forums.developer.nvidia.com/t/cudnn-linking-error-using-cmake/57968/3

TARGET_LINK_LIBRARIES(DeepNeuralNetworkCU
  cudnn
  # It may be that cudnn requires cublas.
  CUDA::cublas)

ADD_EXECUTABLE(GetCuDNNVersion
  GetCuDNNVersion_main.cu
  )

TARGET_LINK_LIBRARIES(GetCuDNNVersion
  DeepNeuralNetworkCU
  UtilitiesCU
  )

SET_TARGET_PROPERTIES(GetCuDNNVersion
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/DeepNeuralNetwork")
