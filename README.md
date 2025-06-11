# Fast-SCNN: Fast Semantic Segmentation Network
A PyTorch implementation of [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/pdf/1902.04502) from the paper by Rudra PK Poudel, Stephan Liwicki.

<p align="center"><img width="100%" src="./png/Fast-SCNN.png" /></p>

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-fast-scnn'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#demo'>Demo</a>
- <a href='#results'>Results</a>
- <a href='#todo'>TO DO</a>
- <a href='#references'>Reference</a>

## Installation
- Python 3.13. Recommended using [uv](https://docs.astral.sh/uv/).
- [PyTorch 2.7](https://pytorch.org/get-started/locally/). Since I use `uv` to manage the dev environment, you can use `uv` to sync the vitural environment later.
- Clone this repository. And change directory into the project folder.
- Run command `uv sync` to sync the same environment as I have. 
- Download the dataset by following the [instructions](#datasets) below.
- Note: For training, we currently support [cityscapes](https://www.cityscapes-dataset.com/), and aim to add [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/).

## Datasets
- You can download [cityscapes](https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Note: please download [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).

## Training-Fast-SCNN
- By default, we assume you have downloaded the cityscapes dataset in the `./datasets/citys` dir.
- To train Fast-SCNN using the train script the parameters listed in `train.py` as a flag or manually change them.
```Shell
python3 train.py --model fast_scnn --dataset citys
# for uv
uv run train.py --model fast_scnn --dataset citys
```

## Evaluation
To evaluate a trained network:
```Shell
python3 eval.py
# for uv
uv run eval.py
```

## Demo
Running a demo:
```Shell
python3 demo.py --model fast_scnn --input-pic './datasets/citys/leftImg8bit/val/lindau/lindau_000012_000019_leftImg8bit.png' --contrast True
# for uv
uv run demo.py --model fast_scnn --input-pic './datasets/citys/leftImg8bit/val/lindau/lindau_000012_000019_leftImg8bit.png' --contrast True
```

## Results
|Method|Dataset|crop_size|mIoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|
|Fast-SCNN(paper)|cityscapes||||
|Fast-SCNN(fork-from)|cityscapes|768|54.84%|92.37%|
|Fast-SCNN(ours)|cityscapes|768|56.64%|93.96%|

Note: The result based on crop_size=768, which is different with paper.

<img src="./png/frankfurt_000001_058914_leftImg8bit.png" width="280" /><img src="./png/frankfurt_000001_058914_gtFine_color.png" width="280" /><img src="./png/frankfurt_000001_058914_seg.png" width="280" />
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(a) test image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b) ground truth &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(c) predicted result

## TODO
- [ ] add distributed training
- [ ] Support for the VOC, ADE20K dataset
- [ ] Support TensorBoard
- [x] save the best model
- [x] add Ohem Loss
 
## Authors
* [**Tramac**](https://github.com/Tramac)

## References
- Rudra PK Poudel. et al. "Fast-SCNN: Fast Semantic Segmentation Network".
