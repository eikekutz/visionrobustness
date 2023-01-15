from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose, ToTensor
import numpy as np

from .augmentations import BrightnessTrans

class Pertubations:
    def __init__(self, steps, model, dataset):
        self.steps = steps
        self.model = model
        self.dataset = dataset
        self.name = "Pertubations"
    def run_test(self):
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False, num_workers=2)
        for step in tqdm(range(self.steps)):
            # adjust pertubation
            transform = self.set_pertubation(step)
            transforms = Compose([transform, ToTensor()])
            # adjust transforms in dataset
            self.dataset.transform = transforms
            
            for data in tqdm(dataloader, total=len(dataloader), desc=f"{self.name} at step {step}"):
                images, labels = data
                images, labels = images, labels
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
            
    def set_pertubation(self, step):
        raise NotImplementedError("set_pertubation not implemented")
        
        
        
        
class Brightness(Pertubations):
    def __init__(self, steps, model, dataset):
        super().__init__(steps, model, dataset)
        self.brightness_values = np.linspace(0, 2, steps)
        self.name = "Brightness"
    def set_pertubation(self, step):
        return BrightnessTrans(self.brightness_values[step])