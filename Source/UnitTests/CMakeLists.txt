ENABLE_TESTING()

SET(TEST_CU_SOURCES
  Activation/Forward_tests.cu
  Activation/ManageDescriptor/ActivationDescriptor_tests.cu
  Activation/ManageDescriptor/SetDescriptor_tests.cu
  Algebra/Modules/Tensors/Tensor4D_tests.cu
  DeepNeuralNetwork/cuDNNLibraryHandle_tests.cu
  Tensors/ManageDescriptor/SetFor4DTensor_tests.cu
  Tensors/ManageDescriptor/TensorDescriptor_tests.cu
  Utilities/DeviceManagement/GetCUDADeviceProperties_tests.cu
  )

ADD_EXECUTABLE(Check
  ${TEST_CU_SOURCES}
  )

TARGET_LINK_LIBRARIES(Check
  ACTIVATION_CU
  DEEPNEURALNETWORK_CU
  TENSORS_CU
  UTILITIES_CU
  gtest_main)

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html
# CUDA_SEPARABLE_COMPILATION enables separate compilation for all CUDA files
# for the given target.
# It should allow us to compile multiple CUDA source files into separate device
# object files and then link them together into a single executable or library.
# It should be necessary when using device code linking (e.g. __device__ or
# __global__ functions in different translation units).
SET_TARGET_PROPERTIES(Check
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")