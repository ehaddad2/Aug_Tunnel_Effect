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
            cifar-10
            cub-200
            flowers-102
            stl-10
            NINCO
            HAM10000

        b. Audio-Based
            
            ESC-50


    2. Predict % OOD performance retained, Pearson Correlation, ID/OOD alignmentbased on probe results => `in progress`

4. Build GB SHAP model, collect all ID/OOD results, and run through it => `not done`
5. Peform analysis based on trained SHAP, if more analysis is needed, use 224 x 224 for those experiments. => `not done`


## Augmentation Details

The following are mappings from the indices of the aug combos and aug polices in the config file to the actual augmentations used in the experiments

1. Basic Augs (applied to all samples)

    0. Resize (256, interpolation=bilinear)
    1. CenterCrop (224|32)
    2. ToTensor()
    3. Normalize(IN-mean, IN-std)

2. Manual Augs
    
    0. Random Horizontal Flip
    1. Random Resize Crop 
    2. Random Affine 
    3. Scale Jitter 
    4. Random Gaussian Blur
    5. Gaussian Noise 
    6. Color Jitter 
    7. Color Distortion 
    8. Random Invert 
    9. Random Solarize
    10. Random Autocontrast
    11. CutOut 
    12. MixUp 
    13. CutMix

3. Aug Polices

    0. SwAV (2 Global Views + 6 Local Views)
    1. Barlow Twins (2 Global Views + 0 Local Views)
    2. DINO (2 Global Views + 6 Local Views)


## Experiment Details & Progress

### 1. Individual aug ablation experiments

[0,0,0,0,0,0,0,0,0,0,0,0,0,0]\=> 
[1,0,0,0,0,0,0,0,0,0,0,0,0,0]\=> 
[0,1,0,0,0,0,0,0,0,0,0,0,0,0]\=>
[0,0,1,0,0,0,0,0,0,0,0,0,0,0]\
[0,0,0,1,0,0,0,0,0,0,0,0,0,0]\
[0,0,0,0,1,0,0,0,0,0,0,0,0,0]\
[0,0,0,0,0,1,0,0,0,0,0,0,0,0]\
[0,0,0,0,0,0,1,0,0,0,0,0,0,0]\
[0,0,0,0,0,0,0,1,0,0,0,0,0,0]\
[0,0,0,0,0,0,0,0,1,0,0,0,0,0]\
[0,0,0,0,0,0,0,0,0,1,0,0,0,0]\
[0,0,0,0,0,0,0,0,0,0,1,0,0,0]\
[0,0,0,0,0,0,0,0,0,0,0,1,0,0]\
[0,0,0,0,0,0,0,0,0,0,0,0,1,0]\
[0,0,0,0,0,0,0,0,0,0,0,0,0,1]

### 2. Kmeans++ init combos
[0]*14                                             \=>
[0.2,0.1,0.1,0.3,0.2,0,0.0,0.3,0.0,0.0,0.3,0,0.3,0]\=> 1
[0.1,0.3,0.1,0.2,0.3,0,0.1,0.0,0.0,0.2,0.0,0.0,0.0,0]\=> 2
[0.0,0.2,0.1,0.0,0.1,0,0.3,0.1,0.1,0.0,0.0,0,0.3,0]\=>3
[0.2,0.3,0.1,0.1,0.2,0,0.0,0.2,0.2,0.1,0.1,0.0,0.1,0]\=>4
[0.0,0.2,0.3,0.0,0.0,0,0.1,0.0,0.0,0.0,0.1,0,0.2,0]\=>5
[0.2,0.3,0.0,0.0,0.0,0,0.0,0.2,0.1,0.2,0.1,0.0,0.3,0]\=>
[0.1,0.2,0.0,0.0,0.2,0,0.2,0.3,0.0,0.2,0.2,0,0.0,0]\=>
[0.0,0.2,0.3,0.3,0.2,0.0,0.1,0.1,0.1,0.3,0.3,0.3,0.3,0.2]\=>
[0.0,0.0,0.0,0.0,0.1,0.1,0.0,0.1,0.0,0.3,0.1,0.2,0.0,0.2]\=>
[0.0,0.3,0.1,0.2,0.0,0.1,0.3,0.3,0.3,0.2,0.0,0.2,0.3,0.0]\=>
[0.2,0.2,0.0,0.1,0.1,0.2,0.3,0.1,0.2,0.1,0.1,0.3,0.1,0.0]\=>
[0.1,0.3,0.2,0.0,0.0,0.1,0.2,0.1,0.1,0.3,0.0,0.0,0.3,0.1]\=>
[0.0,0.1,0.0,0.1,0.0,0.0,0.3,0.3,0.1,0.3,0.3,0.2,0.3,0.2]\=>
[0.1,0.1,0.0,0.3,0.3,0.1,0.1,0.3,0.3,0.0,0.3,0.3,0.1,0.0]=>

### 3. Aug Policies

[1,0,0]\
[0,1,0]\
[0,0,1]
