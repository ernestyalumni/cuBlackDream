#ifndef TRANSFORMER_ATTENTION_PARAMETERS_H
#define TRANSFORMER_ATTENTION_PARAMETERS_H

#include <cudnn.h>
#include <initializer_list>

namespace Transformer
{
namespace Attention
{

enum class AttentionMode : unsigned
{
  //----------------------------------------------------------------------------
  /// Forward declaration of mapping between Q and K, V vectors when beam size
  /// is greater than 1 in the Q input. Multiple Q vectors from same beam bundle
  /// map to same K, V vectors. This means beam sizes in K, V sets are equal to
  /// one.
  //----------------------------------------------------------------------------
  query_map_all_to_one = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE,
  query_map_one_to_one = CUDNN_ATTN_QUERYMAP_ONE_TO_ONE,
  disable_projected_biases = CUDNN_ATTN_DISABLE_PROJ_BIASES,
  //----------------------------------------------------------------------------
  /// Use extra biases in attention input and output projections. In this case,
  /// projected Kbar vectors are computed as Kbar_i = W_K,i K + b * [1,1,..1]_
  /// 1xn, where n is number of columns in K matrix, i.e. same column vector b
  /// is added to all columns of K after weight matrix multiplication.
  //----------------------------------------------------------------------------
  enable_projected_biases = CUDNN_ATTN_ENABLE_PROJ_BIASES
};

//------------------------------------------------------------------------------
/// Bitwise OR-ed flags to attention options. See Supported attnMode flags table
/// for list of supported flags in 7.2.43. cudnnSetAttnDescriptor().
//------------------------------------------------------------------------------
unsigned bitwise_or_attention_options(std::initializer_list<unsigned> inputs);
unsigned bitwise_or_attention_options(
  std::initializer_list<AttentionMode> inputs);

struct Parameters
{
  Parameters(
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
    );

  //----------------------------------------------------------------------------
  /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetAttnDescriptor
  /// \ref 7.2.43. cudnnSetAttnDescriptor()
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  /// Enables various attention options that don't require additional numerical
  /// values. User should assign a preferred set of bitsize OR-ed flags to this
  /// argument.
  //----------------------------------------------------------------------------
  unsigned attention_mode_options_;

  int number_of_attention_heads_;    

  //----------------------------------------------------------------------------
  /// Softmax smooth (1.0 >= smScalar >= 0.0) or sharpening (smScaler > 1.0)
  /// coefficient. Negative values aren't accepted.
  //----------------------------------------------------------------------------
  double sm_scaling_;

  //----------------------------------------------------------------------------
  /// Data type used to represent attention inputs, attention weights, and
  /// attention outputs.
  /// CUDNN_DATA_FLOAT
  //----------------------------------------------------------------------------
  cudnnDataType_t data_type_;

  cudnnDataType_t compute_precision_;

  //----------------------------------------------------------------------------
  /// NVIDIA Tensor Core settings.
  //----------------------------------------------------------------------------
  cudnnMathType_t math_type_;

  //----------------------------------------------------------------------------
  /// Q, K, V embedding vector lengths.
  //----------------------------------------------------------------------------
  int q_size_;
  int k_size_;
  int v_size_;

  //----------------------------------------------------------------------------
  /// Q, K, V embedding vector lengths after input projections. Use 0 to disable
  /// corresponding projection.
  //----------------------------------------------------------------------------
  int q_projected_size_;
  int k_projected_size_;
  int v_projected_size_;

  //----------------------------------------------------------------------------
  /// The h_i vector length after output projection. Use 0 to disable this
  /// projection.
  //----------------------------------------------------------------------------
  int output_projected_size_;

  //----------------------------------------------------------------------------
  /// Largest sequence length expected in sequence data descriptors related to
  /// Q, O, dQ, and dO inputs and outputs.
  //----------------------------------------------------------------------------
  int qo_maximum_sequence_length_;

  //----------------------------------------------------------------------------
  /// Largest sequence length expected in sequence data descriptors related to
  /// K, V, dK, dV
  //----------------------------------------------------------------------------
  int kv_maximum_sequence_length_;

  //----------------------------------------------------------------------------
  /// Largest batch size expected.
  //----------------------------------------------------------------------------
  int maximum_batch_size_;

  //----------------------------------------------------------------------------
  /// Largest beam size expected; beam search scheme can handle multiple q
  /// candidates.
  //----------------------------------------------------------------------------
  int maximum_beam_size_;
};

} // namespace Attention
} // namespace Transformers

#endif // TRANSFORMER_ATTENTION_PARAMETERS_H