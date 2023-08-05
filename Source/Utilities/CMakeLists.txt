SET(UTILITIES_CU_SOURCES
  CuDNNDataTypeToSize.cu
  DeviceManagement/GetCUDADeviceProperties.cu
  ErrorHandling/HandleUnsuccessfulCudaCall.cu
  ErrorHandling/HandleUnsuccessfulCuDNNCall.cu
  )

ADD_LIBRARY(UtilitiesCU
  ${UTILITIES_CU_SOURCES}
  )