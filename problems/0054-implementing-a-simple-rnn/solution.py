import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    Wx = torch.tensor(Wx)
    Wh = torch.tensor(Wh)
    input_sequence = torch.tensor(input_sequence)
    initial_hidden_state = torch.tensor(initial_hidden_state)
    b = torch.tensor(b)

    for x in input_sequence:
        final_hidden_state = torch.tanh(Wx @ x + Wh @ initial_hidden_state + b)
        initial_hidden_state = final_hidden_state

    return final_hidden_state