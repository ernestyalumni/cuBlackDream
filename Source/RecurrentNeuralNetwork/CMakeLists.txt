SET(RECURRENT_NEURAL_NETWORK_CU_SOURCES
  ManageDescriptor/DataDescriptor.cu
  ManageDescriptor/Descriptor.cu
  ManageDescriptor/DropoutDescriptor.cu
  ManageDescriptor/HiddenDescriptor.cu
  ManageDescriptor/InputDescriptor.cu
  ManageDescriptor/LibraryHandleDropoutRNN.cu
  ManageDescriptor/OutputDescriptor.cu
  ManageDescriptor/SetDataDescriptor.cu
  ManageDescriptor/SetDropoutDescriptor.cu
  ManageDescriptor/SetRNNDescriptor.cu
  Modules/GetWeightsAndBias.cu
  Modules/Weight.cu
  Parameters.cu
  SequenceLengthArray.cu
  WeightSpace.cu
  WorkAndReserveSpaces.cu
  )

ADD_LIBRARY(RecurrentNeuralNetworkCU
  ${RECURRENT_NEURAL_NETWORK_CU_SOURCES}
  )

# https://forums.developer.nvidia.com/t/cudnn-linking-error-using-cmake/57968/3

TARGET_LINK_LIBRARIES(RecurrentNeuralNetworkCU
  cudnn
  )
