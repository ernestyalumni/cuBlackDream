#ifndef RECURRENT_NEURAL_NETWORK_SEQUENCE_LENGTH_ARRAY_H
#define RECURRENT_NEURAL_NETWORK_SEQUENCE_LENGTH_ARRAY_H

#include "Parameters.h"

#include <cstdint>
#include <cudnn.h>
#include <vector>

namespace RecurrentNeuralNetwork
{

class HostSequenceLengthArray
{
  public:

    HostSequenceLengthArray(const Parameters& parameters);

    ~HostSequenceLengthArray();

    void set_all_to_maximum_sequence_length();

    void copy_values(const std::vector<int>& input_values);

    int32_t* sequence_length_array_;

  private:

    int batch_size_;
    int maximum_sequence_length_;
};

//------------------------------------------------------------------------------
/// \ref 7.2.47. cudnnSetRNNDataDescriptor()
/// seqLengthArray - an integer array with batchSize number of elements.
/// Describes the length (number of time-steps) of each sequence. Each element
/// must be greater than or equal to 0 but less than or equal to maxSeqLength.
/// In packed layout, elements should be sorted in descending order, similar to
/// the layout required by non-extended RNN compute functions.
//------------------------------------------------------------------------------
class SequenceLengthArray
{
  public:

    SequenceLengthArray(const Parameters& parameters);

    ~SequenceLengthArray();

    void copy_host_input_to_device(const HostSequenceLengthArray& array);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward
    /// \ref 7.2.36.cudnnRNNForward()
    /// devSeqLengths. Input. A copy of seqLengthArray from xDesc or yDesc RNN
    /// data descriptor. devSeqLengths array must be stored in GPU memory as it
    /// is accessed asynchronoously by GPU kernels, possibly after
    /// cudnnRNNForward() function exists.
    //--------------------------------------------------------------------------
    int32_t* sequence_length_array_;

  private:

    int batch_size_;
    int maximum_sequence_length_;
};

} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_SEQUENCE_LENGTH_ARRAY_H