import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torchvision
import collections
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg

class Models:
    def __init__(self):
        self.model_architectures = {
            "resnet10": self.resnet10,
            "resnet18": self.resnet18,
            "vgg13": self.vgg13,
            "vgg19": self.vgg19,
            'lp1': self.lp1,
            'vit_tiny': self.vit_tiny,
            'vit_tiny_plus': self.vit_tiny_plus
        }
        
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
    
    def vgg13(self, num_classes):
        """Initialize vgg13 architecture."""
        layer_setup = [64, 64, 128, 128, 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        return VGG(cfg=layer_setup, class_num=num_classes)

    def vgg19(self, num_classes):
        """Initialize vgg19 architecture."""
        layer_setup = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        return VGG(cfg=layer_setup, class_num=num_classes)
    
    def vit_tiny(self, num_classes):
        """Initialize ViT-Tiny architecture (depth=12)."""
        model = VisionTransformer(
            patch_size=16, 
            embed_dim=192, 
            depth=12, 
            num_heads=3, 
            mlp_ratio=4, 
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes
        )
        model.default_cfg = _cfg()
        return model

    def vit_tiny_plus(self, num_classes):
        """Initialize ViT-Tiny+ architecture (depth=18)."""
        model = VisionTransformer(
            patch_size=8, 
            embed_dim=192, 
            depth=18, 
            num_heads=3, 
            mlp_ratio=4, 
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes
        )
        model.default_cfg = _cfg()
        return model
    def lp1(self, backbone:nn.Module, img_dims, probe_layer:str, probe_out): #performs 1x1 adaptive avg pooling then attaches linear head after probe_layer
        layers = collections.OrderedDict()
        for name, layer in backbone.named_children():
            layers[name] = layer
            if name == probe_layer:break

        partial_backbone = nn.Sequential(layers)
        with torch.no_grad(): probe_in = partial_backbone(torch.randn(1,3,img_dims,img_dims)).shape[1]
        probe = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LinearClassifier(probe_in, probe_out))
        
        layers['probe'] = probe
        new_model = nn.Sequential(layers)
        return new_model
    
    def get_model(self, architecture, num_classes):
        """Retrieve the specified model architecture"""
        assert architecture.lower() in self.model_architectures, f"Unsupported architecture: {architecture}"
        model = self.model_architectures[architecture.lower()](num_classes)
        return model


"""
Helper classes & Functions
"""

def print_model(model:nn.Module):
    print(model)
    print("\nTrainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"{name} (trainable)")       
        else: print(f"{name} (frozen)")


def get_all_probe_layer_names(model_name):
    """Gets all layer names from nn.module, returns them as list of strings"""
    mock_model = Models().get_model(model_name, 10)
    if 'resnet' in model_name:
        return [name for name, module in mock_model.named_modules() if 'conv' in name]
    elif 'vgg' in model_name:
        return [f"features.{i}" for i, module in enumerate(mock_model.features) if isinstance(module, nn.Conv2d)]
    elif 'vit' in model_name:
        return [f"blocks.{i}" for i in range(mock_model.depth)]


class ResNet10(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet10, self).__init__(
            block=BasicBlock, #type of block to make
            layers=[1, 1, 1, 1],  #number of blocks per layer
            num_classes=num_classes)

class VGG(nn.Module):
    def __init__(self, cfg, class_num=100):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, class_num)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                nn.init.kaiming_normal_(conv2d.weight, mode="fan_out", nonlinearity="relu")
                layers += [
                    conv2d,
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)
    
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0, std=0.01)
        self.linear.bias.data.zero_()
        

    def forward(self, x):
        return self.linear(x)  