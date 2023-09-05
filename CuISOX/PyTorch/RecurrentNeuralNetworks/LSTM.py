import torch
import torch.nn as nn

"""
https://pytorch.org/docs/stable/generated/torch.nn.Module.html
torch.nn.Module is the base class for all neural network modules. Your models
should subclass this class.
"""
class EncodingLSTM(nn.Module):
	"""
	@ref https://www.kaggle.com/code/tientd95/understanding-attention-in-neural-network
	@details Used in the following context of an attention network -
	LSTM as encoder produces hidden states for each element in input sequence.
	"""
	def __init__(
		self,
		input_size,
		hidden_size,
		n_layers=1,
		drop_probability=0.0
		):
		super(EncodingLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		"""
		https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.Embedding.html?highlight=embedding

    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    class torch.nn.Embedding(num_embeddings,embedding_dim,padding_idx=None,...)
    @param num_embeddings (int) - size of dictionary of embeddings
    @param embedding_dim (int) - szie of each embedding vector
    @param padding_idx (int, optional) - if specified, entries at padding_idx
    don't contribute to gradient

    EY: 20230902 - num_embeddings is the size of the "vocabulary" or
    "dictionary". embedding_dim is the dimension of the resulting real-valued
    vector. It's observed that the weights are randomly generated, each, element
    according to N(0, 1) (normal distribution) and is a matrix of size
    num_embeddings x embedding_dim
	  """
		self.embedding = nn.Embedding(input_size, hidden_size)

    """
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
    Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.
    @param input_size-number of expected features in input x
    @param hidden_size-number of features in hidden state h
    @param num_layers-number of recurrent layers; e.g. setting num_layers=2
    means stacking 2 LSTMs together to form a stacked LSTM, with second LSTM and
    computing final results. Default: 1.
    """
		self.lstm = nn.LSTM(
			hidden_size,
			hidden_size,
			n_layers,
			dropout=drop_probability,
			batch_first=True)

		def forward(self, inputs, hidden):
			# Embed input words.
			embedded = self.embedding(inputs)

			"""
			https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm
			
			"""

			# Pass embedded word vectors into LSTM and return all outputs
			output, hidden = self.lstm(embedded, hidden)
			return output, hidden

class DecodingLSTM(nn.Module):
	"""
	@ref https://www.kaggle.com/code/tientd95/understanding-attention-in-neural-network
	"""

	def __init__(
		self,
		hidden_size,
		output_size,
		attention,
		n_layers=1,
		drop_probability=0.1
		):
		super(DecodingLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.drop_probability = drop_probability

		# The Attention layer is defined in a separate class
		self.attention = attention

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.dropout = nn.Dropout(self.drop_probability)
		self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
		self.classifier = nn.Linear(self.hidden_size * 2, self.output_size)

  def forward(self, inputs, hidden, encoder_outputs):
      # Embed input words
      embedded = self.embedding(inputs).view(1, 1, -1)
      embedded = self.dropout(embedded)

      # Step 2: Generate new hidden state for decoder
      lstm_out, hidden = self.lstm(embedded, hidden)

      # Step 3: Calculating Alignment Scores
      alignment_scores = self.attention(lstm_out, encoder_outputs)

      # Step 4: Softmaxing alignment scores to obtain Attention weights
      attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)