SET(UTILITIES_CU_SOURCES
  DeviceManagement/GetCUDADeviceProperties.cu
  ErrorHandling/HandleUnsuccessfulCudaCall.cu
  ErrorHandling/HandleUnsuccessfulCuDNNCall.cu
  )

ADD_LIBRARY(UTILITIES_CU
  ${UTILITIES_CU_SOURCES}
  )