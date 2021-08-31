## Expected dataset structure for few shot benchmarks
Dataset for few-shot object detection benchmarks (modified from [TFA](https://github.com/ucbdrive/few-shot-object-detection)). 
The few-shot dataset files can be downloaded [here](https://github.com/Megvii-BaseDetection/GFSD/releases/download/few_shot_split_files/dataset.zip). 
They should have the following directory structure:

### Pascal VOC:
```
voc/
  few_shot_split/
    box_{1,2,3,5,10}shot_{category}_train.txt
    seed{1-29}/
      # shots
```

```
vocfsdet_20{07,12}_trainval_{base,all}{1,2,3}        # Train/val datasets with base categories or all
                                                  categories for splits 1, 2, and 3.
vocfsdet_2007_trainval_all{1,2,3}_{1,2,3,5,10}shot   # Balanced subsets for splits 1, 2, and 3 containing
                                                  1, 2, 3, 5, or 10 shots for each category. You only
                                                  need to specify 2007, as it will load in both 2007
                                                  and 2012 automatically.
vocfsdet_2007_trainval_novel{1,2,3}_{1,2,3,5,10}shot # Same as previous datasets, but only contains data
                                                  of novel categories.
vocfsdet_2007_test_{base,novel,all}{1,2,3}           # Test datasets with base categories, novel categories,
                                                  or all categories for splits 1, 2, and 3.
```

### COCO:
```
coco
  few_shot_split/
    datasplit/
      trainvalno5k.json
      5k.json
    full_box_{1,2,3,5,10,30}shot_{category}_trainval.json
    seed{1-9}/
      # shots
```

Dataset names for config files:
```
cocofsdet_trainval_{base,all}                        # Train/val datasets with base categories or all
                                                  categories.
cocofsdet_trainval_all_{1,2,3,5,10,30}shot           # Balanced subsets containing 1, 2, 3, 5, 10, or 30
                                                  shots for each category.
cocofsdet_trainval_novel_{1,2,3,5,10,30}shot         # Same as previous datasets, but only contains data
                                                  of novel categories.
cocofsdet_test_{base,novel,all}                      # Test datasets with base categories, novel categories,
                                                  or all categories.
```
