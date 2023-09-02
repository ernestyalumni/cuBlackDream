from CuISOX.PyTorch.RecurrentNeuralNetworks.LSTM import LSTM

def test_LSTM_constructs():

	lstm_encoding = LSTM(69, 42)

	assert lstm_encoding.hidden_size == 42