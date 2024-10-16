# Boosting Few-shot Fine-grained Recognition with Background Suppression and Foreground Alignment (BSFA-FSFG)

> **Authors:** 
> Zican Zha,
> [**Hao Tang**](https://scholar.google.com/citations?hl=zh-CN&user=DZXShkoAAAAJ),
> [Yunlian Sun](https://scholar.google.com/citations?user=ObAJh4IAAAAJ&hl=zh-CN),
> and [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN).

This repository provides code for "_**Boosting Few-shot Fine-grained Recognition with Background Suppression and Foreground Alignment**_" IEEE TCSVT 2023. [![IEEE Page](https://img.shields.io/badge/IEEE-TCSVT.2023.3236636-green?style=flat-square)](https://ieeexplore.ieee.org/abstract/document/10018260) [![Arxiv Page](https://img.shields.io/badge/Arxiv-2210.01439-red?style=flat-square)](https://arxiv.org/abs/2210.01439)


## Requirements

 - `Python 3.6`
 - [`Pytorch`](http://pytorch.org/) >= 1.7.0 
 - `Torchvision` = 0.10
 - `scikit-image` = 0.18.1


## Data Preparation

> Download Datasets from [Baidu Drive](https://pan.baidu.com/s/1Bevdjvf5xjroy3U-DA6w7Q) (extraction code: ZZC3)  
> Download Datasets from [Google Drive](https://drive.google.com/file/d/1hK3JwmaM28UnX_yNwm6sGbisCXhF7gJ7/view?usp=sharing)

## How to run

```bash

python train.py --dataset [type of dataset] --model [backbone] --num_classes [num-classes] --nExemplars [num-shots]
python test.py --dataset CUB_200_2011 --model R --num_classes 100 --nExemplars 5

# Example: run on CUB_200_2011 dataset, ResNet-12 backbone, 5-way 1-shot
python train.py --dataset CUB_200_2011 --model R --num_classes 100 --nExemplars 1
python test.py --dataset CUB_200_2011 --model R --num_classes 100 --nExemplars 1

```

## Citation
Please cite our paper if you find the work useful, thanks!
  ```bibtex
     @article{zha2023boosting,
       title={Boosting Few-shot Fine-grained Recognition with Background Suppression and Foreground Alignment},
       author={Zha, Zican and Tang, Hao and Sun, Yunlian and Tang, Jinhui},
       journal={IEEE Transactions on Circuits and Systems for Video Technology},
       year={2023},
       publisher={IEEE},
       doi={10.1109/TCSVT.2023.3236636}
    }
  ```


## Other related papers 

  ```bibtex
     @article{TangYLT22,
          author    = {Hao Tang and Chengcheng Yuan and Zechao Li and Jinhui Tang},
          title     = {Learning attention-guided pyramidal features for few-shot fine-grained recognition},
          journal   = {Pattern Recognit.},
          volume    = {130},
          pages     = {108792},
          year      = {2022}
     }
  ```

## Acknowledgement

This code is based on the implementations of [**fewshot-CAN**](https://github.com/blue-blue272/fewshot-CAN).


