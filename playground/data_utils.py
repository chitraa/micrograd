import torch
from torch.utils.data import Dataset
import os

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device  

def get_input_output_karpathy():
    xs = torch.Tensor([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ])
    ys = torch.Tensor([1.0, -1.0, -1.0, 1.0])
    return xs, ys

def get_input_output():
    xs = torch.Tensor([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [2.0, 1.0, 0.0],
        [-3.0, -1.0, -1.0],
    ])
    # output using random_math_function
    ys = torch.Tensor([0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
    return xs, ys

class TinyDataset(Dataset):
    def __init__(self):
        self.xs, self.ys = get_input_output()

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.xs[idx], self.ys[idx]

def get_tb_logging_dir(log_dir: str):
    logging_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
    return logging_dir

def seed_experiment():
    seed = 42
    torch.manual_seed(seed)
    return seed

def random_math_function(input_array):
    """
    Generate a random mathematical function that takes a 3-dimensional input array
    and outputs a discrete value (-1, 0, or 1).

    Parameters:
    - input_array: Input array of shape (3,).

    Returns:
    - discrete_output: Discrete output value (-1, 0, or 1).
    """
    # Assume a simple random mathematical function for illustration
    result = (input_array[0] * 2 - input_array[1] - input_array[2]**2) / 3.0

    # Apply thresholding to convert to discrete values
    if result > 0.4:
        discrete_output = 1.0
    elif result < 0:
        discrete_output = -1.0
    else:
        discrete_output = 0.0

    return discrete_output

if __name__ == "__main__":

    xs, ys = get_input_output()
    for x in xs: 
        y = random_math_function(x)
        print(y)


