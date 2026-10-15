import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	input_sequence = np.array(input_sequence)
	initial_hidden_state = np.array(initial_hidden_state)

	for x in input_sequence:
		final_hidden_state = np.tanh(Wx @ x + Wh @ initial_hidden_state + b)
		initial_hidden_state = final_hidden_state
	return final_hidden_state