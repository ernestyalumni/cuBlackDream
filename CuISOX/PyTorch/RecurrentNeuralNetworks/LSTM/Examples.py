import torch
import torch.nn as nn

class LSTMWithLinearOutput(nn.Module):
  """
  @ref https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch
  """

  def __init__(
    self,
    input_dim,
    hidden_dim,
    layer_dim,
    output_dim,
    sequence_length,
    learning_rate=0.1):
    """
    @param layer_dim-same as num_layers or number of recurrent layers.
    @param output_dim-Unrelated to LSTM. This is the size of the result after
    applying a linear transform at the end.
    """
    super(LSTMWithLinearOutput, self).__init__()

    # Number of features for a single data point
    self.input_dim = input_dim

    # Hidden dimensions
    self.hidden_dim = hidden_dim
    
    # Number of hidden_layers
    self.layer_dim = layer_dim

    self.sequence_length = sequence_length

    # LSTM
    """
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
    @param input_size-number of expected features in input x
    @param hidden_size-number of features in hidden state h
    @param num_layers-Number of recurrent layers
    @param batch_first-If True, then input and output tensors are provided as
    (batch,seq,feature) instead of (seq, batch, feature)

    All weights and biases initialized from U(-sqrt(k),sqrt(k)) where
    k=1/hidden_size
    """
    self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

    # Readout layer
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
    Class torch.nn.Linear(in_features,out_features,bias=True,..)
    Applies linear transformation to incoming data: y = xA^T + b
    @param in_features(int)-size of each input sample
    @param out_features(int)-size of each output sample

    Variables: weight-learnable weights of module of shape (out_features,
    in_features), values initialized from U(-sqrt(k),sqrt(k)), where
    k=1/in_features. EY: U stands for uniform, I guess.
    bias.
    """
    self.fc = nn.Linear(hidden_dim, output_dim)

    self.loss_and_optimizer = LSTMWithLinearOutput.LossAndOptimizer(
      self.parameters(),
      learning_rate)

  def forward(self, x):
    """
    @ref https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=forward#torch.nn.Module.forward
    @brief Defines computation performed at every call.
    @details This defines every function call by this instance. Should be
    overridden by all subclasses.
    """

    # Initialize hidden state with zeros.
    h0 = torch.zeros(
      # (L, D, H)
      self.layer_dim,
      x.size(0),
      self.hidden_dim).requires_grad_()

    # Initialize cell state
    c0 = torch.zeros(
      # (L, D, H)
      self.layer_dim,
      x.size(0),
      self.hidden_dim).requires_grad_()
    
    # We need to detach as we are doing truncated backpropagation through
    # time. If we don't, we'll backpropagate all the way to start even after
    # going through another batch.
    # (N, L, H_in) when batch_first=True, containing features of input sequence.
    out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

    # Index hidden state of last time step.
    # out.size() --> 100, 28, 100
    # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
    out = self.fc(out[:, -1, :])
    # out.size() --> 100, 10
    return out
  
  def run_on_image_batch(self, image_batch, labels):
    """
    @return outputs, loss
    outputs are the result of "forward" operation by the network, in this case
    LSTM + linear transformation.
    """

    # Load images as a torch tensor with gradient accumulation abilities.
    # https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html?highlight=requires_grad_#torch.Tensor.requires_grad_
    # Change if autograd should record operations on this tensor; sets tensor's
    # requires_grad attirbute in-place. Returns this tensor.
    # requires_grad_()'s main use case is to tell autograd to begin recording
    # operations on a Tensor tensor.
    image_batch_transformed = image_batch.view(
      -1,
      self.input_dim,
      self.sequence_length).requires_grad_()

    # Clear gradients with respect to parameters.
    self.loss_and_optimizer.optimizer.zero_grad()

    # Forward pass to get output/logits.
    # outputs.size 100, 10, i.e. hidden_dim, output_dim
    outputs = self.forward(image_batch_transformed)

    # Calculate Loss: softmax --> cross entropy loss
    loss = self.loss_and_optimizer.cross_entropy_loss(outputs, labels)

    # Getting gradients.
    loss.backward()

    # Updating parameters
    self.loss_and_optimizer.optimizer.step()

    return outputs, loss

  class LossAndOptimizer:
    def __init__(self, model_parameters, learning_rate = 0.1):
      self.cross_entropy_loss = nn.CrossEntropyLoss()
      self.optimizer = torch.optim.SGD(model_parameters, lr=learning_rate)
