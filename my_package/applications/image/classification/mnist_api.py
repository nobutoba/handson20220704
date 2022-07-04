from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

Label = Union[Dict[str, float], str, int, float]


class MNISTInferenceAPI:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.output_size = 10

    def inference(self, input_img_np: np.ndarray) -> Label:
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if input_img_np is None:
            return {i: 0.0 for i in range(10)}  # type: ignore
        input_img_tensor = data_transforms(input_img_np)
        # input_img_tensor = input_img_tensor.view(1, -1)
        input_img_tensor = input_img_tensor.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(input_img_tensor)[0]
            pred = F.softmax(pred, dim=-1)
        return {i: float(pred[i]) for i in range(pred.shape[-1])}  # type: ignore
