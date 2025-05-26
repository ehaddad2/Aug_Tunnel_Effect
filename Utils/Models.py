import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torchvision
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg


class BackboneModel:
    class ResNet10(ResNet):
        def __init__(self, num_classes=1000):
            super(BackboneModel.ResNet10, self).__init__(
                block=BasicBlock, #type of block to make
                layers=[1, 1, 1, 1],  #number of blocks per layer
                num_classes=num_classes)

    class VGG(nn.Module):
        def __init__(self, cfg, class_num=100):
            super(BackboneModel.VGG, self).__init__()
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

    """
    Module for creating backbones based on supported architectures:
    - resnet10
    - resnet18  
    - vgg13
    - vgg19
    - vit_tiny
    - vit_tiny_plus
    """
    
    def __init__(self):
        self.model_architectures = {
            "resnet10": self.resnet10,
            "resnet18": self.resnet18,
            "vgg13": self.vgg13,
            "vgg19": self.vgg19,
            'vit_tiny': self.vit_tiny,
            'vit_tiny_plus': self.vit_tiny_plus
        }
        
    def resnet10(self, num_classes):
        """Initialize ResNet10 architecture."""
        model = BackboneModel.ResNet10(num_classes=num_classes)
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
        return BackboneModel.VGG(cfg=layer_setup, class_num=num_classes)

    def vgg19(self, num_classes):
        """Initialize vgg19 architecture."""
        layer_setup = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        return BackboneModel.VGG(cfg=layer_setup, class_num=num_classes)
    
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
    
    def load_backbone(self, path, architecture, num_classes):
        """Loads a pretrained backbone from a model dict"""
        backbone = self.get_model(architecture, num_classes)
        backbone.load_state_dict(torch.load(path, map_location='cpu'))
        return backbone


    def get_model(self, architecture, num_classes):
        """Retrieve the specified model architecture"""
        assert architecture.lower() in self.model_architectures, f"Unsupported architecture: {architecture}"
        model = self.model_architectures[architecture.lower()](num_classes)
        return model
    
class CNNProbe(nn.Module):
    def __init__(self, backbone, probe_layer, num_classes, img_dims):
        super().__init__()
        self.backbone = backbone
        if hasattr(self.backbone, 'classifier'): self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'): self.backbone.fc = nn.Identity()
        self.backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        self.probe_layer = probe_layer
        self.probe_output = None
        self._register_hook()
        
        with torch.no_grad(): #obtain dim info
            dummy_input = torch.randn(1, 3, img_dims, img_dims)
            _ = self.backbone(dummy_input)
            
            if self.probe_output is None:
                raise ValueError(f"Layer {probe_layer} not found in the model")
            
            pooled = nn.AdaptiveAvgPool2d((2, 2))(self.probe_output)
            feature_size = pooled.view(-1).shape[0]
        
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Linear(feature_size, num_classes)
        self.classifier.weight.data.normal_(mean=0, std=0.01)
        self.classifier.bias.data.zero_()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            self.probe_output = output
        
        #find module to attach the hook
        for name, module in self.backbone.named_modules():
            if name == self.probe_layer:
                module.register_forward_hook(hook_fn)
                break
    
    def forward(self, x):
        self.probe_output = None
        self.backbone(x) #probe result stored in probe_output
        
        if self.probe_output is None:
            raise RuntimeError(f"No features captured from layer: {self.probe_layer}")
        
        x = self.pool(self.probe_output)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class ViTProbe(nn.Module):
    def __init__(self, backbone, probe_layer, num_classes):
        super().__init__()
        self.backbone = backbone
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        self.backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        self.probe_layer = probe_layer
        self.probe_features = None
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)
        self._register_hook()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            #skip the class token and average the patch tokens
            if output.dim() == 3 and output.size(1) > 1:  #shape = [B, N, C]
                self.probe_features = output[:, 1:].mean(dim=1)  #GAP over patch tokens
            else:
                self.probe_features = output
        
        #look for probing layer, attach hook
        for name, module in self.backbone.named_modules():
            if name == self.probe_layer:
                module.register_forward_hook(hook_fn)
                break
    
    def forward(self, x):
        self.backbone(x)
        return self.classifier(self.probe_features)
    

"""
Helper classes & Functions
"""

def print_model(model:nn.Module):
    print(model)
    print("\nTrainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"{name} (trainable)")       
        else: print(f"{name} (frozen)")
    
def get_all_probe_layer_names(args):
    backbone_name = args.backbone_architecture.lower()
    for key, layers in args.preset_probing_layers.items():
        if backbone_name in key.lower():
            return layers
    print(f"Warning: No predefined layers for architecture '{backbone_name}'. Using empty list.")
    return []

if __name__ == '__main__':
    import sys
    # Test parameters
    img_dims = 224
    num_classes = 10
    batch_size = 2
    
    print("=== Testing Probe Model Creation for Multiple Architectures ===\n")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_dims, img_dims)
    
    # 1. Test with ResNet18
    print("1. Testing ResNet18 Probing")
    backbone_model = BackboneModel()
    resnet = backbone_model.resnet18(num_classes=num_classes)
    
    # Define layers to probe for ResNet18
    resnet_probe_layers = [
        "conv1",
        "layer1.0.conv1",
        "layer1.0.conv2",
        "layer1.1.conv1",
        "layer1.1.conv2",
        "layer2.0.conv1",
        "layer2.0.conv2",
        "layer2.1.conv1",
        "layer2.1.conv2",
        "layer3.0.conv1",
        "layer3.0.conv2",
        "layer3.1.conv1",
        "layer3.1.conv2",
        "layer4.0.conv1",
        "layer4.0.conv2",
        "layer4.1.conv1",
        "layer4.1.conv2"
    ]
    
    print(f"\nTesting {len(resnet_probe_layers)} ResNet18 layers:")
    for layer in resnet_probe_layers:
        try:
            probe_model = CNNProbe(resnet, layer, num_classes, img_dims)
            
            # Test forward pass
            with torch.no_grad():
                output = probe_model(dummy_input)
            
            print(f"  ✓ Layer '{layer}' - Output shape: {output.shape}")
        except Exception as e:
            print(f"  ✗ Layer '{layer}' - Error: {e}")
    
    print("\n" + "="*80 + "\n")
    
    # 2. Test with VGG19
    print("2. Testing VGG19 Probing")
    vgg = backbone_model.vgg19(num_classes=num_classes)

    # Define layers to probe for VGG19
    vgg_probe_layers = [
        "features.0",
        "features.3",
        "features.7",
        "features.10",
        "features.14",
        "features.17",
        "features.21",
        "features.24",
        "features.28",
        "features.31"
    ]
    
    print(f"\nTesting {len(vgg_probe_layers)} VGG19 layers:")
    for layer in vgg_probe_layers:
        try:
            probe_model = CNNProbe(vgg, layer, num_classes, img_dims)
            
            # Test forward pass
            with torch.no_grad():
                output = probe_model(dummy_input)
            
            print(f"  ✓ Layer '{layer}' - Output shape: {output.shape}")
        except Exception as e:
            print(f"  ✗ Layer '{layer}' - Error: {e}")
    
    print("\n" + "="*80 + "\n")
    
    # 3. Test with ViT-Tiny
    print("3. Testing ViT-Tiny Probing")
    vit = backbone_model.vit_tiny(num_classes=num_classes)
    
    # Define layers to probe for ViT-Tiny
    vit_probe_layers = [
        "patch_embed",
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
        "blocks.4",
        "blocks.5",
        "blocks.6",
        "blocks.7",
        "blocks.8",
        "blocks.9",
        "blocks.10",
        "blocks.11",
        "norm"
    ]
    
    print(f"\nTesting {len(vit_probe_layers)} ViT-Tiny layers:")
    for layer in vit_probe_layers:
        try:
            probe_model = ViTProbe(vit, layer, num_classes)
            
            # Test forward pass
            with torch.no_grad():
                output = probe_model(dummy_input)
            
            print(f"  ✓ Layer '{layer}' - Output shape: {output.shape}")
        except Exception as e:
            print(f"  ✗ Layer '{layer}' - Error: {e}")
    
    print("\nProbe testing completed successfully!")