"""Main module."""
import torch
import torch.nn as nn
from .perturbations import Brightness

class TestPertubations:
    def __init__(self):
        self.pertubations = {
            "brightness": Brightness
        }
    def run_test(self, model, dataset, steps):
        for pertubation in self.pertubations:
            pertubation = self.pertubations[pertubation](steps, model, dataset)
            pertubation.run_test()
