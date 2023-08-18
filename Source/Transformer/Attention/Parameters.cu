#include "Parameters.h"

#include <cudnn.h>
#include <initializer_list>
#include <stdexcept>

using std::runtime_error;

namespace Transformer
{
namespace Attention
{

unsigned bitwise_or_attention_options(std::initializer_list<unsigned> inputs)
{
  unsigned result {0};

  for (const auto input : inputs)
  {
    result = result | input;
  }

  return result;
}

unsigned bitwise_or_attention_options(
  std::initializer_list<AttentionMode> inputs)
{
  unsigned result {0};

  for (const auto input : inputs)
  {
    result = result | static_cast<unsigned>(input);
  }

  return result;
}

Parameters::Parameters(
  const unsigned attnMode,
  const int nHeads,
  const double smScalar,
  cudnnDataType_t dataType,
  cudnnDataType_t computePrec,
  cudnnMathType_t mathType,
  const int qSize,
  const int kSize,
  const int vSize,
  const int qProjSize,
  const int kProjSize,
  const int vProjSize,
  const int oProjSize,
  const int qoMaxSeqLength,
  const int kvMaxSeqLength,
  const int maxBatchSize,
  const int maxBeamSize
  ):
  attention_mode_options_{attnMode},
  number_of_attention_heads_{nHeads},
  sm_scaling_{smScalar},
  data_type_{dataType},
  compute_precision_{computePrec},
  math_type_{mathType},
  q_size_{qSize},
  k_size_{kSize},
  v_size_{vSize},
  q_projected_size_{qProjSize},
  k_projected_size_{kProjSize},
  v_projected_size_{vProjSize},
  output_projected_size_{oProjSize},
  qo_maximum_sequence_length_{qoMaxSeqLength},
  kv_maximum_sequence_length_{kvMaxSeqLength},
  maximum_batch_size_{maxBatchSize},
  maximum_beam_size_{maxBeamSize}
{
  if ((qSize <= 0) ||
    (kSize <= 0) ||
    (vSize <= 0))
  {
    throw runtime_error("Q, K, V must be positive");
  }

  if ((qProjSize < 0) ||
    (kProjSize < 0) ||
    (vProjSize < 0))
  {
    throw runtime_error("Q, K, V projected sizes cannot be negative");
  }

  if (0.0 > smScalar)
  {
    throw runtime_error("Negative values not accepted for smScalar");
  }

  if (qProjSize != kProjSize)
  {
    throw runtime_error(
      "Projected sizes for Q and K must be equal for dot product to work.");
  }

  // From Table 50. Supported Combinations for cudnnSetAttnDescriptor().
  if (dataType == CUDNN_DATA_DOUBLE)
  {
    if ((computePrec != CUDNN_DATA_DOUBLE) || (mathType != CUDNN_DEFAULT_MATH))
    {
      throw runtime_error("Invalid data or math types");
    }
  }

  if (dataType == CUDNN_DATA_FLOAT)
  {
    if ((computePrec != CUDNN_DATA_FLOAT) || (
      (mathType != CUDNN_DEFAULT_MATH) &&
        (mathType != CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)))
    {
      throw runtime_error("Invalid data or math types");
    }
  }

  if (dataType == CUDNN_DATA_HALF)
  {
    if ((
      (computePrec != CUDNN_DATA_HALF) && (computePrec != CUDNN_DATA_FLOAT)) ||
        ((mathType != CUDNN_DEFAULT_MATH) &&
          (mathType != CUDNN_TENSOR_OP_MATH) &&
          (mathType != CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)))
    {
      throw runtime_error("Invalid data or math types");
    }
  }
}

} // namespace Attention
} // namespace Transformer