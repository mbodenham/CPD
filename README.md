# Cascaded Partial Decoder for Fast and Accurate Salient Object Detection (CVPR2019)

# Changelog
## 1.0 - 2019-05-26
### Added
- vgg.py - Pytorch Hub to fetch pre-trained VGG16 model
- test_CPD.py - Use CUDA only if available
- test_CPD.py - Progress print out

### Changed

- test_CPD.py - Corrected dataset paths

### Fixed
- test_CPD.py - Replaced deprecated torch.nn.functional.upsample with torch.nn.functional.interpolate
- test_CPD.py - Replaced deprecated scipy.misc.imsave with torchvision.utils.save_image
- CPD_models.py - Relative import in Python 3
- CPD_ResNet_models.py - Relative import in Python 3



# Documentation
Our model ranks first in the challenging [SOC benchmark](http://dpfan.net/SOCBenchmark/) up to now (2019.11.6).

## Requirements:

Python 3.8.2, PyTorch 1.5.0

## Usage

Modify the paths of backbone and datasets, then run test_CPD.py

## Pre-trained model

VGG16     backbone: [google drive](https://drive.google.com/open?id=1ddopz30_sNPOb0MvTCoNwZwL-oQDMGIW), [BaiduYun](https://pan.baidu.com/s/18qF_tpyRfbgZ0YLleP8c5A) (code: gb5u)

ResNet50  backbone: [google drive](https://drive.google.com/open?id=188sybU9VU5rW2BH2Yzhko4w-G5sPp6yG), [BaiduYun](https://pan.baidu.com/s/1tc6MWlj5sbMJJGCyUNFxbQ) (code: klfd)

## Pre-computed saliency maps

VGG16     backbone: [google drive](https://drive.google.com/open?id=1LcCTcKGEsZjO8WUgbGpiiZ4atQrK1u_O)

ResNet50  backbone: [google drive](https://drive.google.com/open?id=16pLY2qYZ1KIzPRwR7zFUseEDJiwhdHOg)

## Performance

Maximum F-measure

|Model|FPS|ECSSD|HKU-IS|DUT-OMRON|DUTS-TEST|PASCAL-S|
|:----|:----|:----|:----|:----|:----|:----|
|PiCANet|7|0.931|0.921|0.794|0.851|0.862|
|CPD|66|0.936|0.924|0.794|0.864|0.866|
|PiCANet-R|5|0.935|0.919|0.803|0.860|0.863|
|CPD-R|62|0.939|0.925|0.797|0.865|0.864|

MAE

|Model|ECSSD|HKU-IS|DUT-OMRON|DUTS-TEST|PASCAL-S|
|:----|:----|:----|:----|:----|:----|
|PiCANet|0.046|0.042|0.068|0.054|0.076|
|CPD|0.040|0.033|0.057|0.043|0.074|
|PiCANet-R|0.046|0.043|0.065|0.051|0.075|
|CPD-R|0.037|0.034|0.056|0.043|0.072|

## Shadow Detection

pre-computed maps: [google drive](https://drive.google.com/open?id=1R__w0FXpMhUMnIuoxPaX6cFzwAypX13U)

## Performance

BER

|Model|SBU|ISTD|UCF|
|:----|:----|:----|:----|
|DSC|5.59|8.24|8.10|
|CPD|4.19|6.76|7.21|

# Citation
```
@InProceedings{Wu_2019_CVPR,
author = {Wu, Zhe and Su, Li and Huang, Qingming},
title = {Cascaded Partial Decoder for Fast and Accurate Salient Object Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
