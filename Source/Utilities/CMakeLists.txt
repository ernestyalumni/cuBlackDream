SET(UTILITIES_CU_SOURCES
  CuDNNDataTypeToSize.cu
  DeviceManagement/GetCUDADeviceProperties.cu
  ErrorHandling/HandleUnsuccessfulCudaCall.cu
  ErrorHandling/HandleUnsuccessfulCuDNNCall.cu
  )

SET(UTILITIES_SOURCES
  Time/WrangleDates.cpp
  )

ADD_LIBRARY(UtilitiesCU
  ${UTILITIES_CU_SOURCES}
  ${UTILITIES_SOURCES}
  )