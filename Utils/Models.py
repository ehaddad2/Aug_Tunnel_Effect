import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torchvision
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torchinfo import summary

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
    
    
class lp1(nn.Module):
    def __init__(self, dim, num_classes=100):
        super(lp1, self).__init__()
        self.linear = nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x) 

class FeatureExtractor:
    def __init__(self, backbone, layer_names, cache_dir="./features_cache", model_arch=None, model_type="cnn"):
        self.backbone = backbone
        self.layer_names = layer_names
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.features_cache = {}
        self.activations = {}
        self.model_arch = model_arch or type(backbone).__name__
        self.model_type = model_type.lower()
        
        if hasattr(self.backbone, 'classifier'): self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'): self.backbone.fc = nn.Identity()
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.hooks = self.register_hooks()
        if torch.cuda.is_available(): self.backbone = self.backbone.cuda()

    def register_hooks(self):
        hooks = {}
        def _get_activation(layer):
            def hook(module, input, output): #prep hooks
                if self.model_type == "cnn":
                    pooled = F.adaptive_avg_pool2d(output, (1, 1))
                    self.activations[layer] = torch.flatten(pooled, 1).detach()

                elif self.model_type == "vit": #TODO: verify
                    if len(output.shape) == 3:  # ViT output [B, N+1, D]
                        # Extract image tokens (exclude class token)
                        image_tokens = output[:, 1:, :]
                        # Apply global average pooling
                        self.activations[layer] = torch.mean(image_tokens, dim=1).detach()
                    else:
                        self.activations[layer] = output.detach()
            return hook
        
        for name, module in self.backbone.named_modules(): #attach hooks
            if name in self.layer_names:
                hooks[name] = module.register_forward_hook(_get_activation(name))
                
        return hooks

    def check_features_exist(self, dataset_name, split):
        feature_dir = self.cache_dir / self.model_arch / dataset_name / split

        if not feature_dir.exists():
            return False
        return True
    
    def save_features(self, features, targets, dataset_name, split):
        feature_dir = self.cache_dir / self.model_arch / dataset_name / split
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        for k in features:
            data_dict = {'features': features[k], 'targets': targets} #pair layerwise features with ood ds target
            feature_file = feature_dir / f"{k}.pt"
            torch.save(data_dict, feature_file)
            print(f'Saved features at {feature_file}')
            
            self.features_cache[f"{split}_{dataset_name}_{k}"] = data_dict
    
    def extract_features(self, dataloader, dataset_name, split='train', force_recompute=False):

        if not force_recompute and self.check_features_exist(dataset_name, split):
            print(f"Features for {dataset_name} ({split}) already exist. Loading from cache.")
            self.load_features(dataset_name, split)
            return
        
        features = {layer: [] for layer in self.layer_names}
        all_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f"Extracting {split} features"):

                if torch.cuda.is_available(): inputs = inputs.cuda()
                self.activations = {}
                _ = self.backbone(inputs)
                
                for layer in self.layer_names:
                    features[layer].append(self.activations[layer].cpu())
                
                all_targets.append(targets)
        
        concatenated_features = {}
        for layer in self.layer_names:
            concatenated_features[layer] = torch.cat(features[layer], 0)
        
        all_targets = torch.cat(all_targets, 0)
        self.save_features(concatenated_features, all_targets, dataset_name, split)
    
    def load_features(self, dataset_name, split):
        feature_dir = self.cache_dir / self.model_arch / dataset_name / split
        
        for layer_name in self.layer_names:
            feature_file = feature_dir / f"{layer_name}.pt"
            if feature_file.exists():
                data = torch.load(feature_file)
                self.features_cache[f"{split}_{dataset_name}_{layer_name}"] = data
    
    def get_features(self, dataset_name, layer_name, split='train'):
        cache_key = f"{split}_{dataset_name}_{layer_name}"
        if cache_key not in self.features_cache:
            self.load_features(dataset_name, split)

        return self.features_cache.get(cache_key, None)

        
"""
Helper classes & Functions
"""

def print_model(model:nn.Module, input_size=(1, 3, 224, 224)):
    summary(model, input_size)
    
def get_all_probe_layer_names(args):
    backbone_name = args.backbone_architecture.lower()
    for key, layers in args.preset_probing_layers.items():
        if backbone_name in key.lower():
            return layers
    print(f"Warning: No predefined layers for architecture '{backbone_name}'. Using empty list.")
    return []

arch_to_probe = {
    'lp1': lp1
}