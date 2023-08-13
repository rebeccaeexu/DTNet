# DTNet
Direction-aware Video Demoiréing with Temporal-guided Bilateral Learning



### Requirements

* basicsr==1.4.2
* scikit-image==0.15.0



### Dataset Preparation

* Download dataset from [Data-V1](https://www.dropbox.com/sh/5pkntn20785gqqj/AADmYCroOu5YDhzGam6Nhoz9a?dl=0) and [Data-V2](https://www.dropbox.com/sh/7trmzm2slm2qlg8/AADt3e8MH_52EyLKFtZwXirJa?dl=0). Unzip and copy the files to  `data`.
* Download and unzip our pre-trained model, and copy it to `experiments/Train_DTNet_resume.yml/models`.

Organize the directories as follows:

```
┬─ experiments
│   └─ Train_DTNet_resume.yml
│		└─ models
│       	├─ DTNet_f.pth
│     	 	└─ DTNet_g.pth
└─ data
    ├─ homo
    │   ├─ iphone
    │   │   ├─ train
    |   |   |	├─ source
    |   |   |	|	└─ ... (image filename)
	│   |   | 	└─ target
	│   |   |		└─ ... (corresponds to the former)
	│   | 	└─ test
	│   |		└─ ..
    |   |
    │   └─ tcl
    │       └─ ... 
    └─ of
        ├─ iphone
        │   └─ ... 
        └─ tcl
            └─ ... 
```



### How to Test

* Download the pre-trained model

* Example: Testing on the TCL-V2 dataset

```python
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/Test_DTNet.yml
```



### How to train

* Single GPU training

```python
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/Train_DTNet_scratch.yml
```

* Distributed training

```python
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/Train_DTNet_scratch.yml --launcher pytorch
```