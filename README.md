# [FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation (ICCV 2023)](https://github.com/li-ronghui/FineDance)

## Teaser

<img src="teaser/teaser.png">

## Dataset

### Download

The dataset can be downloaded at [Google Drive](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view?usp=sharing)

### Dataset split

We spilt FineDance dataset into train, val and test sets in two ways: FineDance@Genre and  FineDance@Dancer. Each music and paired dance are only present in one split. 

1. The test set of FineDance@Genre includes a broader range of dance genres, but the same dancer appear in  train/val/test set. Although the training set and test set include the same dancers, the same motions do not appear in both the training and testing sets. This is because these dancers do not have distinct personal characteristics in their dances.
2. The train/val/test set of FineDance@Dancer was divided by different dancers, which test set contains fewer dance genres, yet the same dancer won't appear in different sets.

If you use this dataset for dance generation, we recommend you to use the split of FineDance@Genre.