import torch
import torch.nn as nn

class Attention(nn.Module):
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
		super(Attention, self).__init__()
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