import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torchvision
import collections

class Models:
    def __init__(self, device):
        self.model_architectures = {
            "resnet10": self.resnet10,
            "resnet18": self.resnet18,
            'lp1': self.lp1,
        }
        if device.type != 'xla': self.device = device
    def __del__(self):
        torch.cuda.empty_cache()
        
    def resnet10(self, num_classes):
        """Initialize ResNet10 architecture."""
        model = ResNet10(num_classes=num_classes)
        return model

    def resnet18(self, num_classes):
        """Initialize ResNet18 architecture."""
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        return model

    def lp1(self, backbone:nn.Module, probe_layer:str, probe_in:int, probe_out:int): #performs 1x1 adaptive avg pooling then attaches linear head after probe_layer
        layers = collections.OrderedDict()
        for name, layer in backbone.named_children():
            layers[name] = layer
            if name == probe_layer:break

        probe = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LinearClassifier(probe_in, probe_out)
        )
        layers['probe'] = probe
        new_backbone = nn.Sequential(layers)
        return new_backbone
    
    def get_model(self, architecture, num_classes):
        """Retrieve the specified model architecture"""
        assert architecture.lower() in self.model_architectures, f"Unsupported architecture: {architecture}"
        model = self.model_architectures[architecture.lower()](num_classes)
        return model
class ResNet10(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet10, self).__init__(
            block=BasicBlock, #type of block to make
            layers=[1, 1, 1, 1],  #number of blocks per layer
            num_classes=num_classes)

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0, std=0.01)
        self.linear.bias.data.zero_()
        

    def forward(self, x):
        return self.linear(x)  

"""
Helper Functions
"""

def print_model(model:nn.Module):
    print(model)
    print("\nTrainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"{name} (trainable)")       
        else: print(f"{name} (frozen)")