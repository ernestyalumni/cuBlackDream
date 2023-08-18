#ifndef TRANSFORMER_MANAGE_DESCRIPTOR_LIBRARY_HANDLE_DROPOUTS_ATTENTION_H
#define TRANSFORMER_MANAGE_DESCRIPTOR_LIBRARY_HANDLE_DROPOUTS_ATTENTION_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "Transformer/Attention/Parameters.h"
#include "Transformer/ManageDescriptor/AttentionDescriptor.h"

namespace Transformer
{

namespace ManageDescriptor
{

struct LibraryHandleDropoutsAttention
{
  LibraryHandleDropoutsAttention(
    const Transformer::Attention::Parameters& parameters,
    const float attention_dropout_probability = 0,
    const unsigned long long attention_seed = 1337ull,
    const float post_dropout_probability = 0,
    const unsigned long long post_seed = 1337ull
    );

  ~LibraryHandleDropoutsAttention() = default;

  DeepNeuralNetwork::CuDNNLibraryHandle handle_;
  RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor
    attention_dropout_descriptor_;
  RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor
    post_dropout_descriptor_;

  AttentionDescriptor descriptor_;

  float attention_dropout_probability_;
  float post_dropout_probability_;
  unsigned long long attention_seed_;
  unsigned long long post_seed_;
};

} // namespace ManageDescriptor
} // namespace Transformer

#endif // TRANSFORMER_MANAGE_DESCRIPTOR_LIBRARY_HANDLE_DROPOUTS_ATTENTION_H