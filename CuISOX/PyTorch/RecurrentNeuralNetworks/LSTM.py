import torch
import torch.nn as nn

"""
https://pytorch.org/docs/stable/generated/torch.nn.Module.html
torch.nn.Module is the base class for all neural network modules. Your models
should subclass this class.
"""
class LSTM(nn.Module):
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
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		"""
		https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.Embedding.html?highlight=embedding

		
		"""
		self.embedding = nn.Embedding(input_size, hidden_size)
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
