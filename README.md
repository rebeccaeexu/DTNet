# DTNet
**[AAAI-2024] Direction-aware Video Demoiréing with Temporal-guided Bilateral Learning**

[Paper Link](https://arxiv.org/abs/2308.13388)



### Requirements

* basicsr==1.4.2
* scikit-image==0.15.0
* deepspeed



### Prepare

1. **Download [VDMoire dataset](https://github.com/CVMI-Lab/VideoDemoireing).**
2. **Download the [pretrined models](https://www.dropbox.com/scl/fi/sfiypguwynxwfvghjoeij/experiments.zip?rlkey=n6wr63odbmnlhh6ydavpag2kd&dl=0).**

Organize the directories as follows:

```
┬─ experiments
│   └─ Train_DTNet_resume_ipv1.yml
│	│	└─ models
│   │    	├─ DTNet_f.pth
│   │  	 	└─ DTNet_g.pth
│   └─ Train_DTNet_resume_ipv2.yml
│   └─ Train_DTNet_resume_tclv1.yml
│   └─ Train_DTNet_resume_tclv2.yml
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

* Example: Testing on the TCL-V2 dataset

```python
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/Test_DTNet_tclv2.yml
```



### How to train

* Single GPU training

```python
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/Train_DTNet_scratch_ipv1.yml
```

* Distributed training

```python
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/Train_DTNet_scratchipv1.yml --launcher pytorch
```

