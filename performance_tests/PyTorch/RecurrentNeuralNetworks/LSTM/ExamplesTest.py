# Setup paths
from pathlib import Path
import sys
file_ancestor_path = Path(__file__).resolve().parent.parent.parent.parent.parent
print("File's ancestor or 'base' path --- ", file_ancestor_path)
print("sys.path before changes: ", sys.path)
if str(file_ancestor_path) not in sys.path:
	sys.path.append(str(file_ancestor_path))

from CuISOX.DataWrangling.Kaggle.DigitRecognizer import ProcessDigitsData
from CuISOX.PyTorch.RecurrentNeuralNetworks.LSTM.Examples import (
    LSTMWithLinearOutput,
    train_LSTMWithLinearOutput_model_on_images)
from CuISOX.utilities.configure_paths import DataPaths
from CuISOX.utilities.DataIO.KagglePaths import KagglePaths

import time

def main():
	data_paths = DataPaths()
	kaggle_paths = KagglePaths()
	kaggle_data_file_paths = kaggle_paths.get_all_data_file_paths()
	digit_paths = kaggle_data_file_paths["DigitRecognizer"]
	training_data_paths = DataPaths.get_path_with_substring(digit_paths, "train")
	training_data_path = data_paths.Kaggle() / training_data_paths[0]

	process_digits_data = ProcessDigitsData()
	process_digits_data.parse_csv(training_data_path)
	process_digits_data.load_data()

	input_dim = 28
	hidden_dim = 100
	layer_dim = 1
	output_dim = 10
	sequence_length = 28
	# EY: 20230906, we'll use one of the physical dimensions (I'm guessing
	# width) as the "sequence" variable. So we imagine that each row is an input
	# and each successive row makes a sequence of rows of pixels of the image.
	lstm_with_linear_output = LSTMWithLinearOutput(
		input_dim,
		hidden_dim,
		layer_dim,
		output_dim,
		sequence_length)	

	batch_size = 100
	# Originally, this value was 6000.
	n_iters = 20000

	number_of_epochs = process_digits_data.calculate_epoch(n_iters, batch_size)

	start_training_time = time.time()

	lost_list, iteration_list, accuracy_list = \
		train_LSTMWithLinearOutput_model_on_images(
	    	lstm_with_linear_output,
	    	process_digits_data,
	    	number_of_epochs)

	training_time_duration = time.time() - start_training_time
	print("--- %s seconds ---" % training_time_duration)

	print("Final loss, final accuracy: ", lost_list[-1].item(), accuracy_list[-1])

if __name__ == "__main__":
	main()