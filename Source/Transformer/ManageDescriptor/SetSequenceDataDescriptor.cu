#include "SetSequenceDataDescriptor.h"

#include <algorithm>
#include <cstddef>
#include <cudnn.h>
#include <vector>

namespace Transformer
{
namespace ManageDescriptor
{

SetSequenceDataDescriptor::CreateDimensionsAndSequenceLengths::
  CreateDimensionsAndSequenceLengths(
    const int time_dimension,
    const int batch_dimension,
    const int beam_dimension,
    const int vector_dimension
    ):
    dimensions_array_{},
    sequence_length_array_{},
    sequence_length_array_total_size_{0}
{
  dimensions_array_[CUDNN_SEQDATA_TIME_DIM] = time_dimension;
  dimensions_array_[CUDNN_SEQDATA_BATCH_DIM] = batch_dimension;
  dimensions_array_[CUDNN_SEQDATA_BEAM_DIM] = beam_dimension;
  dimensions_array_[CUDNN_SEQDATA_VECT_DIM] = vector_dimension;

  sequence_length_array_total_size_ =
    dimensions_array_[CUDNN_SEQDATA_BATCH_DIM] *
      dimensions_array_[CUDNN_SEQDATA_BEAM_DIM];

  sequence_length_array_.reserve(sequence_length_array_total_size_);
  sequence_length_array_.resize(sequence_length_array_total_size_);
  std::fill(sequence_length_array_.begin(), sequence_length_array_.end(), 0);
}

void SetSequenceDataDescriptor::CreateDimensionsAndSequenceLengths::
  set_sequence_length_array(const std::vector<int>& input_sequence_lengths)
{
  for (std::size_t i {0}; i < sequence_length_array_total_size_; ++i)
  {
    const int sequence_length {input_sequence_lengths.at(i)};
    if (sequence_length <= dimensions_array_[CUDNN_SEQDATA_TIME_DIM])
    {
      sequence_length_array_[i] = sequence_length;
    }
  }
}

void SetSequenceDataDescriptor::CreateDimensionsAndSequenceLengths::
  fill_with_maximum_sequence_length()
{
  std::fill(
    sequence_length_array_.begin(),
    sequence_length_array_.end(),
    dimensions_array_[CUDNN_SEQDATA_TIME_DIM]);
}

} // namespace ManageDescriptor
} // namespace Transformer
