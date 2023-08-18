#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "SetAttentionDescriptor.h"
#include "Transformer/Attention/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor;
using Transformer::Attention::Parameters;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace Transformer
{
namespace ManageDescriptor
{

HandleUnsuccessfulCuDNNCall set_attention_descriptor(
  AttentionDescriptor& descriptor,
  const Parameters& parameters,
  DropoutDescriptor& attention_dropout_descriptor,
  DropoutDescriptor& post_dropout_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_set_descriptor {
    "Failed to set attention descriptor"};

  handle_set_descriptor(cudnnSetAttnDescriptor(
    descriptor.descriptor_,
    parameters.attention_mode_options_,
    parameters.number_of_attention_heads_,
    parameters.sm_scaling_,
    parameters.data_type_,
    parameters.compute_precision_,
    parameters.math_type_,
    attention_dropout_descriptor.descriptor_,
    post_dropout_descriptor.descriptor_,
    parameters.q_size_,
    parameters.k_size_,
    parameters.v_size_,
    parameters.q_projected_size_,
    parameters.k_projected_size_,
    parameters.v_projected_size_,
    parameters.output_projected_size_,
    parameters.qo_maximum_sequence_length_,
    parameters.kv_maximum_sequence_length_,
    parameters.maximum_batch_size_,
    parameters.maximum_beam_size_));

  return handle_set_descriptor;
}

HandleUnsuccessfulCuDNNCall get_attention_descriptor(
  AttentionDescriptor& descriptor,
  Parameters& parameters,
  DropoutDescriptor& attention_dropout_descriptor,
  DropoutDescriptor& post_dropout_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_get_descriptor {
    "Failed to get attention descriptor"};

  handle_get_descriptor(cudnnGetAttnDescriptor(
    descriptor.descriptor_,
    &(parameters.attention_mode_options_),
    &(parameters.number_of_attention_heads_),
    &(parameters.sm_scaling_),
    &(parameters.data_type_),
    &(parameters.compute_precision_),
    &(parameters.math_type_),
    &(attention_dropout_descriptor.descriptor_),
    &(post_dropout_descriptor.descriptor_),
    &(parameters.q_size_),
    &(parameters.k_size_),
    &(parameters.v_size_),
    &(parameters.q_projected_size_),
    &(parameters.k_projected_size_),
    &(parameters.v_projected_size_),
    &(parameters.output_projected_size_),
    &(parameters.qo_maximum_sequence_length_),
    &(parameters.kv_maximum_sequence_length_),
    &(parameters.maximum_batch_size_),
    &(parameters.maximum_beam_size_)));

  return handle_get_descriptor;
}

} // namespace ManageDescriptor
} // namespace Transformer