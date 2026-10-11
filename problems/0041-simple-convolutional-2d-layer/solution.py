import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	pad_input = None
	if padding != 0:
		pad_input = np.pad(input_matrix, pad_width = padding, mode = 'constant')
	else:
		pad_input = input_matrix
	output_height = (input_height + 2*padding - kernel_height)//stride + 1
	output_width = (input_width + 2*padding - kernel_width)//stride + 1
	output = np.zeros((output_height, output_width))
	for out_x in range(output_height):
		for out_y in range(output_width):
			start_x = out_x * stride
			start_y = out_y * stride
			output[out_x][out_y] = np.sum(pad_input[start_x:start_x + kernel_height, start_y: start_y + kernel_width] * kernel).item()
	return output
