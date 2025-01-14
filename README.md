# Impact of Augmentations on Tunnel Effect

A project building off of findings from https://arxiv.org/abs/2405.15018

Usage
-------------

### Training Backbone + Probes

1. cd into 'OOD' directory
2. modify the args.json file to set the desired parameters for training
3. run the following command to train the backbone model and/or probes:
```bash
python main.py --args_file args.json
```

### Analysis
in development

## TODO:

1. Setup wandb => `done`
2. Figure out backbone experiments (aug combos)

    1. aug combos => `done`
    3. scale down to 32 x 32 => `done`
    4. test out and analyze results on some manual augs/aug polices (record aug configs as inputs into SHAP) => `in progress`
    5. add in more architectures (VGG, DenseNet, etc) `not done`

3. Re-run probes => `in progress`

    1. more OOD datasets (eventually audio)
        
        a. Image-Based
            
            aircrafts
            cifar10
            cub200
            flowers102
            stl10
            NINCO
            Oxford IIIT Pet
            HAM10000

        b. Audio-Based
            
            ESC-50
            (AudioSet)
            (Common Voice)

    2. Predict % OOD performance retained, Pearson Correlation, ID/OOD alignmentbased on probe results => `in progress`

4. Build GB SHAP model, collect all ID/OOD results, and run through it => `not done`
5. Peform analysis based on trained SHAP, if more analysis is needed, use 224 x 224 for those experiments. => `not done`

## Dataset Details

1. ID Datasets

    1. ImageNet100

2. OOD Datasets

    1. CIFAR10
    2. CUB200
    3. esc-50
    4. STL10
    5. Flowers102
    6. Aircrafts
    7. HAM10000
    8. NINCO

## Augmentation Details

The following are mappings from the indices of the aug combos and aug polices in the config file to the actual augmentations used in the experiments

1. Basic Augs (applied to all samples)

    0. Resize (256, interpolation=bilinear)
    1. CenterCrop (224|32)
    2. ToTensor()
    3. Normalize(IN-mean, IN-std)

2. Manual Augs
    
    0. Random Horizontal Flip (p=0.5)
    1. Random Resize Crop (size=(224|32))
    2. Random Affine (degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)
    3. Scale Jitter (size=(224|32), scale_range=(0.1, 2.0), interpolation=bilinear)
    4. Random Gaussian Blur (p=0.5, radius_min=0.1, radius_max=2.0)
    5. Gaussian Noise (p=0.5, mean=0, sigma=0.1)
    6. Color Jitter (brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    7. Color Distortion (brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    8. Random Invert (p=0.5)
    9. Random Solarize (threshold=128, p=0.5)
    10. Random Autocontrast (p=0.5)
    11. CutOut (n_holes=1, size=(8x8))
    12. MixUp ($\alpha$=1.0)
    13. CutMix ($\beta$=1.0)

3. Aug Polices

    0. SwAV (2 Global Views + 6 Local Views)
    1. Barlow Twins (2 Global Views + 0 Local Views)
    2. DINO (2 Global Views + 6 Local Views)


## Experiment Details & Progress

### 1. Individual aug distangling experiments

[0,0,0,0,0,0,0,0,0,0,0,0,0,0]\=> `done`\
[1,0,0,0,0,0,0,0,0,0,0,0,0,0]\=> `done`\
[0,1,0,0,0,0,0,0,0,0,0,0,0,0]\=> `done`\
[0,0,1,0,0,0,0,0,0,0,0,0,0,0]\=> `done`\
[0,0,0,1,0,0,0,0,0,0,0,0,0,0]\=> `done`\
[0,0,0,0,1,0,0,0,0,0,0,0,0,0]\=> `done`\
[0,0,0,0,0,1,0,0,0,0,0,0,0,0]\=> `done`\
[0,0,0,0,0,0,1,0,0,0,0,0,0,0]\=> `done`\
[0,0,0,0,0,0,0,1,0,0,0,0,0,0]\
[0,0,0,0,0,0,0,0,1,0,0,0,0,0]\
[0,0,0,0,0,0,0,0,0,1,0,0,0,0]\
[0,0,0,0,0,0,0,0,0,0,1,0,0,0]\
[0,0,0,0,0,0,0,0,0,0,0,1,0,0]\
[0,0,0,0,0,0,0,0,0,0,0,0,1,0]\
[0,0,0,0,0,0,0,0,0,0,0,0,0,1]

### 2. 20 Kmeans++ init combos

[0,0,0,1,0,1,1,0,1,1,0,1,1,1]\
[1,0,0,0,0,0,0,1,0,1,0,1,0,0]\
[1,1,1,0,0,1,1,0,0,0,0,1,1,0]\
[1,0,1,1,0,0,1,0,1,0,0,1,1,1]\
[0,0,0,1,1,0,1,1,0,0,0,0,1,0]\
[1,1,1,1,1,0,1,1,0,1,0,1,0,1]\
[1,1,0,0,0,0,0,0,1,1,0,1,0,1]\
[1,1,0,0,0,1,0,0,1,0,1,1,0,1]\
[0,0,1,0,0,1,0,0,0,0,0,0,1,0]\
[0,1,1,0,1,1,1,1,0,1,1,1,1,1]\
[0,1,0,1,1,1,0,1,1,1,1,1,0,0]\
[1,1,1,0,1,1,0,0,1,1,1,1,0,0]\
[1,0,1,0,0,0,0,1,1,0,1,0,0,1]\
[1,1,0,1,0,0,0,0,0,1,1,0,1,1]\
[0,1,1,0,1,1,0,1,0,1,0,0,0,0]\
[0,0,1,1,1,0,0,0,1,0,1,1,1,0]\
[1,0,0,0,1,1,0,0,0,1,1,0,0,0]\
[0,0,0,0,1,1,1,1,0,0,1,0,0,0]\
[1,1,0,1,0,0,1,1,0,0,0,1,1,1]\
[1,0,0,1,1,1,1,0,1,0,1,1,0,0]\

### 3. Aug Polices

[1,0,0]\
[0,1,0]\
[0,0,1]
