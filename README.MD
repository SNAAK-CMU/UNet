# This is my personal UNet Repository

## Steps

### Dataset Preparation:

1. Obtain images as .jpg files and masks as .png files
2. The masks must be single channel masks with the 3rd channel corresponding to class ID. If masks are multichannel, run ./make_single_channel.py by specifying the color and corresponding class ID in __main__
3. Arrange dataset as:
    --dataset/
        --imgs/
        --masks/
4. Create folder Segment_config/ in dataset/ and create 4 files: test.txt, trainval.txt, train.txt and val.txt
5. Set path to dataset/ in ./voc_annotation.py and ensure it runs and populates the .txt files. You can also decide train:val:test split in this file prior to running it


### Training:

1. create ./logs directory
2. set path to dataset/ in train.py
5. check unet.py file and train.py file to set configurations
6. remember to properly set num_classes and the correctly corresponding class ID and color in single channel mask (you need to visualise the mask in an image viewer for this)
6. run train.py and monitor training
7. in root directory. run 
```bash
    tesnorboard --logdir logs/
```
For tensorboard

### Inference:
1. in predict.py set the "mode" accordingly. "predict" is the default mode
2. in unet.py set the model_path, num_classes, input_shape and mix_type as needed
3. run predict.py