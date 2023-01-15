import torch
from PIL import Image, ImageEnhance


class BrightnessTrans(torch.nn.Module):
    """
    Brightness pertubation for fixed value
    """
    def __init__(self, brightness:float):
        super().__init__()
        self.brightness = brightness
    def forward(self, x):
        enhancer = ImageEnhance.Brightness(x)
        return enhancer.enhance(self.brightness)
