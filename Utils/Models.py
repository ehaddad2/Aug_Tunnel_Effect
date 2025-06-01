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
    
class CNNHookProbe(nn.Module):
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


class ViTProbe(nn.Module):
    """
    Sequential-based ViT probe that applies GAP to patch tokens at each layer
    """
    def __init__(self, backbone, probe_layer, num_classes):
        super().__init__()
        self.backbone = backbone
        self.probe_layer = probe_layer
        self.num_classes = num_classes
        
        # Prep backbone
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Create feature extractor
        self.feature_extractor = self._create_feature_extractor()
        
        # Attach probe head
        self._setup_classifier()
    
    def _create_feature_extractor(self):
        """Create ViT feature extractor up to specified layer"""
        layers = []
        
        # Always include patch embedding
        layers.append(self.backbone.patch_embed)
        
        if self.probe_layer == "patch_embed":
            return ViTFeatureExtractor(layers, self.backbone.pos_embed, 
                                     self.backbone.cls_token, self.backbone.pos_drop)
        
        # Add transformer blocks
        if "blocks." in self.probe_layer:
            target_block_idx = int(self.probe_layer.split('.')[-1])
            
            # Add blocks up to and including target block
            for i in range(target_block_idx + 1):
                if i < len(self.backbone.blocks):
                    layers.append(self.backbone.blocks[i])
        
        elif self.probe_layer == "norm":
            # Add all blocks plus norm
            layers.extend(list(self.backbone.blocks))
            layers.append(self.backbone.norm)
        
        return ViTFeatureExtractor(layers, self.backbone.pos_embed, 
                                 self.backbone.cls_token, self.backbone.pos_drop)
    
    def _setup_classifier(self):
        """Setup classifier for ViT features"""
        # ViT features are typically embed_dim size after global average pooling
        feature_size = self.backbone.embed_dim
        self.classifier = nn.Linear(feature_size, self.num_classes)
        self.classifier.weight.data.normal_(mean=0, std=0.01)
        self.classifier.bias.data.zero_()
    
    def forward(self, x):
        # Extract features - returns [B, embed_dim] after GAP over patch tokens
        features = self.feature_extractor(x)
        
        # Classify
        output = self.classifier(features)
        return output


class ViTFeatureExtractor(nn.Module):
    """Helper class to handle ViT feature extraction with proper GAP at each layer"""
    def __init__(self, layers, pos_embed, cls_token, pos_drop):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.pos_embed = pos_embed
        self.cls_token = cls_token
        self.pos_drop = pos_drop
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding (first layer)
        x = self.layers[0](x)  # [B, N, embed_dim] where N is number of patches
        
        # If we're only probing patch_embed, apply GAP immediately
        if len(self.layers) == 1:
            # For patch_embed layer, we have patch tokens without cls token
            # Apply GAP directly to patch tokens
            return x.mean(dim=1)  # [B, embed_dim]
        
        # Add class token for transformer blocks
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply remaining transformer blocks
        for layer in self.layers[1:]:
            x = layer(x)
        
        # Apply GAP to patch tokens (exclude cls token at index 0)
        # x shape: [B, 1+N, embed_dim] where first token is cls token
        patch_tokens = x[:, 1:, :]  # [B, N, embed_dim] - exclude cls token
        features = patch_tokens.mean(dim=1)  # [B, embed_dim] - GAP over patch dimension
        
        return features

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

def test_probe_models():
    """Test all probe models with different architectures and layers"""
    print("="*80)
    print("TESTING PROBE MODELS")
    print("="*80)
    
    # Test parameters
    img_dims = 224
    num_classes = 10
    batch_size = 4
    
    # Probe layer configurations
    probe_configs = {
        "resnet18": [
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
        ],
        "vgg19": [
            "features.0",
            "features.3",
            "features.7",
            "features.10",
            "features.14",
            "features.17",
            "features.20",
            "features.23",
            "features.27",
            "features.30",
            "features.33",
            "features.36",
            "features.40",
            "features.43",
            "features.46",
            "features.49"
        ],
        "vit_tiny": [
            "patch_embed",
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",

        ]
    }
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_dims, img_dims)
    backbone_model = BackboneModel()
    
    # Test each architecture
    for arch_name, probe_layers in probe_configs.items():
        print(f"\n{'='*20} TESTING {arch_name.upper()} {'='*20}")
        
        try:
            # Create backbone
            backbone = backbone_model.get_model(arch_name, num_classes)
            
            # Test each probe layer
            successful_layers = []
            failed_layers = []
            
            for layer in probe_layers:
                try:
                    print(f"\nTesting layer: {layer}")
                    
                    # Create appropriate probe
                    if arch_name.startswith('vit'):
                        probe = ViTProbe(backbone, layer, num_classes)
                        print(f"Probe for layer {layer}: {probe}")
                    else:
                        probe = CNNProbe(backbone, layer, num_classes, img_dims)
                    
                    
                    # Test forward pass
                    with torch.no_grad():
                        output = probe(dummy_input)
                    
                    # Verify output shape
                    expected_shape = (batch_size, num_classes)
                    if output.shape == expected_shape:
                        print(f"  ✓ SUCCESS - Output shape: {output.shape}")
                        successful_layers.append(layer)
                    else:
                        print(f"  ✗ SHAPE MISMATCH - Expected: {expected_shape}, Got: {output.shape}")
                        failed_layers.append(layer)
                        
                except Exception as e:
                    print(f"  ✗ ERROR - {str(e)}")
                    failed_layers.append(layer)
            
            # Summary for this architecture
            print(f"\n{arch_name.upper()} SUMMARY:")
            print(f"  Successful layers: {len(successful_layers)}/{len(probe_layers)}")
            print(f"  Failed layers: {len(failed_layers)}")
            
            if failed_layers:
                print(f"  Failed: {failed_layers}")
            
        except Exception as e:
            print(f"ERROR creating {arch_name} backbone: {e}")
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")


def test_dataparallel_compatibility():
    """Test DataParallel compatibility"""
    print("\n" + "="*80)
    print("TESTING DATAPARALLEL COMPATIBILITY")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping DataParallel test")
        return
    
    if torch.cuda.device_count() < 2:
        print("Less than 2 GPUs available - skipping DataParallel test")
        return
    
    # Test parameters
    img_dims = 224
    num_classes = 10
    batch_size = 8
    
    # Create backbone and probe
    backbone_model = BackboneModel()
    backbone = backbone_model.get_model("resnet18", num_classes)
    
    # Test CNNProbe with DataParallel
    probe = CNNProbe(backbone, "layer1.0.conv1", num_classes, img_dims)
    probe = nn.DataParallel(probe)
    probe = probe.cuda()
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, 3, img_dims, img_dims).cuda()
    
    try:
        with torch.no_grad():
            output = probe(dummy_input)
        print(f"✓ DataParallel CNNProbe SUCCESS - Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ DataParallel CNNProbe FAILED - {e}")
    
    # Test ViTProbe with DataParallel
    vit_backbone = backbone_model.get_model("vit_tiny", num_classes)
    vit_probe = ViTProbe(vit_backbone, "blocks.0", num_classes)
    vit_probe = nn.DataParallel(vit_probe)
    vit_probe = vit_probe.cuda()
    
    try:
        with torch.no_grad():
            output = vit_probe(dummy_input)
        print(f"✓ DataParallel ViTProbe SUCCESS - Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ DataParallel ViTProbe FAILED - {e}")


if __name__ == "__main__":
    # Run tests
    test_probe_models()
    test_dataparallel_compatibility()