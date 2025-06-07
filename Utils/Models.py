import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torchvision
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
import os
from tqdm import tqdm

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
    
class CNNFeatureExtractor:
    def __init__(self, backbone, layer_names, cache_dir="./features_cache", model_arch=None):
        self.backbone = backbone
        self.layer_names = layer_names
        self.cache_dir = cache_dir
        self.features_cache = {}
        self.hook_features = {}
        self.model_arch = model_arch
        
        os.makedirs(cache_dir, exist_ok=True)
        
        if hasattr(self.backbone, 'classifier'): self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'): self.backbone.fc = nn.Identity()
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self._register_hooks()
    
    def _register_hooks(self):
        for layer_name in self.layer_names:
            for name, module in self.backbone.named_modules():
                if name == layer_name:
                    module.register_forward_hook(self._hook_fn(layer_name))
                    break
    
    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            feats = output
            feats = nn.AdaptiveAvgPool2d((2, 2))(feats) #only pool after last layer
            feats = feats.view(feats.size(0), -1)
            
            if layer_name not in self.hook_features:
                self.hook_features[layer_name] = []
            self.hook_features[layer_name].append(feats.cpu().detach())
        return hook
    
    def extract_features(self, dataloader, dataset_name, force_recompute=False):
        dataset_dir = os.path.join(self.cache_dir, self.model_name, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        if not force_recompute:
            all_cached = all(os.path.exists(os.path.join(dataset_dir, f"{layer}.pt")) for layer in self.layer_names)
            if all_cached:
                for layer_name in self.layer_names:
                    cache_path = os.path.join(dataset_dir, f"{layer_name}.pt")
                    with open(cache_path, 'rb') as f:
                        self.features_cache[f"{dataset_name}_{layer_name}"] = torch.load(f)
                return
        
        self.hook_features = {}
        labels = []
        with torch.no_grad(): #run through ood dataset and let hooks store features
            for batch_data, batch_labels in tqdm(dataloader):
                if torch.cuda.is_available():
                    batch_data = batch_data.cuda()
                    self.backbone = self.backbone.cuda()
                
                _ = self.backbone(batch_data)
                labels.append(batch_labels)
        
        labels = torch.cat(labels, dim=0)
        
        for layer_name in self.layer_names: #collect layerwise features and save
            if layer_name in self.hook_features:
                features = torch.cat(self.hook_features[layer_name], dim=0)
                feature_data = {'features': features, 'labels': labels}
                
                cache_path = os.path.join(dataset_dir, f"{layer_name}.pt")
                with open(cache_path, 'wb') as f:
                    torch.save(feature_data, f)
                
                self.features_cache[f"{dataset_name}_{layer_name}"] = feature_data
    
    def get_features(self, dataset_name, layer_name):
        return self.features_cache[f"{dataset_name}_{layer_name}"]
    
class ViTHookProbe(nn.Module):
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
    

class CNNProbe(nn.Module):
    """
    sequential-based CNN probe
    """
    def __init__(self, backbone, probe_layer, num_classes, img_dims):
        super().__init__()
        self.backbone = backbone
        self.probe_layer = probe_layer
        self.num_classes = num_classes
        self.img_dims = img_dims
        
        #prepare backbone
        if hasattr(self.backbone, 'classifier'): 
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'): 
            self.backbone.fc = nn.Identity()
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #build probe
        self.feature_extractor = self._create_feature_extractor()
        self._setup_classifier()
    
    def _create_feature_extractor(self):
        """create sequential feature extractor up to probe layer"""
        
        if hasattr(self.backbone, 'conv1'): #resnet
            return self._create_resnet_extractor()
        
        elif hasattr(self.backbone, 'features'): #vgg
            return self._create_vgg_extractor()
        
        else:
            raise ValueError(f"Unsupported backbone architecture used for CNN probing: {self.backbone}")
    
    def _create_resnet_extractor(self):
        """Create ResNet feature extractor"""
        layers = []
        
        if self.probe_layer == "conv1":
            layers.append(self.backbone.conv1)
            return nn.Sequential(*layers)
        
        layers.extend([ #backbone prologue
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        ])
        
        #unpack 'layer' set
        if "layer" in self.probe_layer:
            parts = self.probe_layer.split('.')
            layer_name, block_idx, conv_name = parts[0], int(parts[1]), parts[2]
            layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
            target_layer_idx = int(layer_name[len(layer_name)-1])-1
            
            for i in range(target_layer_idx): #add previous layers
                if hasattr(self.backbone, layer_names[i]):
                    layers.append(getattr(self.backbone, layer_names[i]))
            
            #add layer block components
            if hasattr(self.backbone, layer_name):
                target_layer = getattr(self.backbone, layer_name)
                
                for i in range(block_idx): #add previous blocks
                    layers.append(target_layer[i])
                
                #add in remaining (partial) block
                target_block = target_layer[block_idx]
                if conv_name == "conv1":
                        layers.append(target_block.conv1)
                elif conv_name == "conv2":
                    layers.extend([
                        target_block.conv1,
                        target_block.bn1,
                        target_block.relu,
                        target_block.conv2
                    ])

            else:
                raise ModuleNotFoundError(f"CNN PROBE ERROR: {layer_name} not found in backbone!")
        return nn.Sequential(*layers)
    
    def _create_vgg_extractor(self):
        target_idx = int(self.probe_layer.split('.')[-1])
        feature_layers = list(self.backbone.features.children())
        layers = feature_layers[:target_idx + 1]
        return nn.Sequential(*layers)

    
    def _setup_classifier(self):
        """connect feature extractor with probe head"""

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.img_dims, self.img_dims)
            if next(self.backbone.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            features = self.feature_extractor(dummy_input)
            
            if len(features.shape) == 4:  #feature shape: [B, C, H, W], ie: we only attach probe after conv layers
                pooled = nn.AdaptiveAvgPool2d((2, 2))(features)
                flattened_size = pooled.view(pooled.size(0), -1).shape[1]

            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        #construct probe head
        classifier = nn.Linear(flattened_size, self.num_classes)
        classifier.weight.data.normal_(mean=0, std=0.01)
        classifier.bias.data.zero_()

        self.probe_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            classifier
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        output = self.probe_head(features)
        return output






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

