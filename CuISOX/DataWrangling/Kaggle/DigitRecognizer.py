from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import torch

class ProcessDigitsData:
  """
  @ref https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch  
  """
  def __init__(self, test_proportion=0.2, random_state=42, batch_size=100):
    self.test_proportion = test_proportion
    self.random_state = random_state
    self.batch_size = batch_size

  @staticmethod
  def parse_csv_no_split(file_path, dtype_input=np.float32):
    data_csv = pd.read_csv(file_path, dtype=dtype_input)

    # Divide by 255 to normalize
    X_numpy = data_csv.loc[:].values / 255

    X_numpy = torch.from_numpy(X_numpy)
    return X_numpy

  @staticmethod
  def load_data_no_split(X_numpy, input_batch_size):
    tensor_data_set = TensorDataset(X_numpy)
    return DataLoader(
      tensor_data_set,
      batch_size=input_batch_size,
      shuffle=False)

  def parse_csv(self, file_path, dtype_input=np.float32):
    training_data_csv = pd.read_csv(file_path, dtype=dtype_input)

    # Split data into features (pixels) and labels (numbers from 0 to 9)
    y_numpy = training_data_csv.label.values
    # Divide by 255 to normalize
    X_numpy = training_data_csv.loc[
      :,
      training_data_csv.columns != "label"].values / 255
    X_train, X_test, y_train, y_test = train_test_split(
      X_numpy,
      y_numpy,
      test_size = self.test_proportion,
      random_state = self.random_state)
    
    # Create feature and targets tensors for training set. We need a variable to
    # accumulate gradients. Therefore, first create tensor, and then we'll
    # create variable.

    self.X_training = torch.from_numpy(X_train)
    # data type is Long
    self.y_training = torch.from_numpy(y_train).type(torch.LongTensor)

    # Create feature and targets tensor for test set.
    self.X_testing = torch.from_numpy(X_test)
    self.y_testing = torch.from_numpy(y_test).type(torch.LongTensor)

  def load_data(self):
    try:
      # Pytorch training and test sets
      training = TensorDataset(self.X_training, self.y_training)
      testing = TensorDataset(self.X_testing, self.y_testing)
    except AttributeError:
      return

    # Data Loader
    self.training_loader = DataLoader(
      training,
      batch_size=self.batch_size,
      shuffle=False)
    
    self.test_loader = DataLoader(
      testing,
      batch_size=self.batch_size,
      shuffle=False)

  def calculate_epoch(self, number_of_iterations, batch_size):
    return int(number_of_iterations / (len(self.X_training) / batch_size))