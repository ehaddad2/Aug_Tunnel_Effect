# Impact of Augmentations on Tunnel Effect

A project building off of findings from https://arxiv.org/abs/2405.15018

Abstract:

Supervised pre-trained deep neural networks (DNNs) contain embeddings that are widely used for downstream classification tasks; However, their performance can vary widely based in part on factors impacting the training dataset. In this paper, we adopt a revised tunnel-effect hypothesis from earlier work, suggesting representation compression is the primary cause of poor out-of-distribution (OOD) generalization. We quantify the impact of varying augmentation strategies and unique training sample sizes using a novel SHAP analysis based on the performance of various linear probes on various OOD datasets. Our results shed light on the effects over-augmentation and large unique sample counts have on OOD generalization.

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


## Experiment Details & Progress (no scale jitter, cutout, cutmix)

### 1. kmeans++ experiments (Model = resnet18)

hyperparam:
lr = 0.01

levels = [0, 0.1, 0.3, 0.5, 0.7] 
0. [0,0,0,0,0,0,0,0,0,0,0,0,0,0] => 1
1. [0.1,0.3,0.7,0,0.0,0.7,0.7,0.1,0.1,0.5,0.7,0,0.1,0] => 1r
2. [0.3,0.7,0.0,0,0.7,0.0,0.7,0.5,0.1,0.3,0.3,0,0.1,0] => 2
3. [0.7,0.7,0.5,0,0.0,0.7,0.5,0.1,0.3,0.7,0.7,0,0.7,0] => 3
4. [0.0,0.7,0.3,0,0.1,0.0,0.1,0.5,0.0,0.1,0.3,0,0.5,0] => 3
5. [0.3,0.3,0.5,0,0.1,0.7,0.3,0.0,0.1,0.3,0.0,0,0.3,0] => 4
6. [0.5,0.3,0.0,0.0,0.1,0.0,0.7,0.7,0.7,0.1,0.3,0.0,0.5,0.0] => 2r
7. [0.3,0.3,0.3,0.0,0.1,0.0,0.1,0.5,0.3,0.0,0.7,0.0,0.1,0.0] => 4
8. [0.7,0.0,0.0,0.0,0.7,0.1,0.0,0.7,0.5,0.0,0.7,0.0,0.3,0.0] => 5
9. [0.1,0.7,0.1,0.0,0.1,0.5,0.7,0.1,0.5,0.7,0.3,0.0,0.3,0.0] => 6
10. [0.0,0.5,0.7,0.0,0.5,0.5,0.7,0.5,0.0,0.1,0.1,0.0,0.1,0.0] => 7

### 2. kmeans++ experiments (Model = vgg19)

hyperparam:
lr = 0.006

levels = [0, 0.1, 0.3, 0.5, 0.7] (no scale jitter, cutout, cutmix)
11. [0,0,0,0,0,0,0,0,0,0,0,0,0,0] => 5
12. [0.1,0.3,0.7,0,0.0,0.7,0.7,0.1,0.1,0.5,0.7,0,0.1,0] => 6
13. [0.3,0.7,0.0,0,0.7,0.0,0.7,0.5,0.1,0.3,0.3,0,0.1,0] => 7r
14. [0.7,0.7,0.5,0,0.0,0.7,0.5,0.1,0.3,0.7,0.7,0,0.7,0] => 1
15. [0.0,0.7,0.3,0,0.1,0.0,0.1,0.5,0.0,0.1,0.3,0,0.5,0] => 6
16. [0.3,0.3,0.5,0,0.1,0.7,0.3,0.0,0.1,0.3,0.0,0,0.3,0] => 3
17. [0.5,0.3,0.0,0.0,0.1,0.0,0.7,0.7,0.7,0.1,0.3,0.0,0.5,0.0] => 4
18. [0.3,0.3,0.3,0.0,0.1,0.0,0.1,0.5,0.3,0.0,0.7,0.0,0.1,0.0] => 5
19. [0.7,0.0,0.0,0.0,0.7,0.1,0.0,0.7,0.5,0.0,0.7,0.0,0.3,0.0] => 6
20. [0.1,0.7,0.1,0.0,0.1,0.5,0.7,0.1,0.5,0.7,0.3,0.0,0.3,0.0] => 1
21. [0.0,0.5,0.7,0.0,0.5,0.5,0.7,0.5,0.0,0.1,0.1,0.0,0.1,0.0] => 3

### 3. kmeans++ experiments (Model = vit-tiny)

hyperparam:
lr = 0.0008
batch size = 96
ep = 40

levels = [0, 0.1, 0.3, 0.5, 0.7] (no scale jitter, cutout, cutmix)
22. [0,0,0,0,0,0,0,0,0,0,0,0,0,0] => 4
23. [0.1,0.3,0.7,0,0.0,0.7,0.7,0.1,0.1,0.5,0.7,0,0.1,0] => 5
24. [0.3,0.7,0.0,0,0.7,0.0,0.7,0.5,0.1,0.3,0.3,0,0.1,0] => 6
25. [0.7,0.7,0.5,0,0.0,0.7,0.5,0.1,0.3,0.7,0.7,0,0.7,0] => 7
26. [0.0,0.7,0.3,0,0.1,0.0,0.1,0.5,0.0,0.1,0.3,0,0.5,0] => 1
27. [0.3,0.3,0.5,0,0.1,0.7,0.3,0.0,0.1,0.3,0.0,0,0.3,0] => 4
28. [0.5,0.3,0.0,0.0,0.1,0.0,0.7,0.7,0.7,0.1,0.3,0.0,0.5,0.0] => 5
29. [0.3,0.3,0.3,0.0,0.1,0.0,0.1,0.5,0.3,0.0,0.7,0.0,0.1,0.0] => 6
30. [0.7,0.0,0.0,0.0,0.7,0.1,0.0,0.7,0.5,0.0,0.7,0.0,0.3,0.0] => 7
31. [0.1,0.7,0.1,0.0,0.1,0.5,0.7,0.1,0.5,0.7,0.3,0.0,0.3,0.0] => 1
32. [0.0,0.5,0.7,0.0,0.5,0.5,0.7,0.5,0.0,0.1,0.1,0.0,0.1,0.0] => 7



If you change batch size due to hardware, you need to scale LR appropriately

for example, if you reduce batch size to 256, LR should be scaled by 0.5 (for VGG: 3e-3) (edited) 

effective LR = base LR x (chosen batch size / 512)
