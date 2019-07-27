# Animating all things
### Columbia Summer '19 COMSW4995 Deep Learning Final Project

This repository contains the source code for the Columbia Summer '19 COMSW4995 Deep Learning final project by [Xipeng Xie](https://github.com/Ageneinair), [Nikita Lockshin](https://github.com/Smthri), [Lianfeng Wang](https://github.com/KnightLian). This project is inspired by Siarohin *et al*'s work in [Monkey-net](http://www.stulyakov.com/papers/monkey-net.html) and [Mask-RCNN](https://github.com/matterport/Mask_RCNN).

### Installation

To install the dependencies run:
```
pip install -r requirements.txt
```


### Motion Transfer Demo 

To run a demo, download a [checkpoint]() and run the following command:
```
python demo.py --config config/moving-gif.yaml --checkpoint <path/to/checkpoint>
```
The result will be stored in ```demo.gif```.


### Visualization of the Process
```
python demo.py --i_am_iddo_drori True --config config/moving-gif.yaml --checkpoint <path/to/checkpoint>
```


### Training

To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder.
To check the loss values during training in see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.


### Datasets

1) **Shapes**. This dataset is saved along with repository.
Training takes about 17 minuts in Colab.

2) **Actions**. This dataset is also saved along with repository.
 And training takes about 1 hour.

3) **Taichi**. We havn't tried this.

4) **MGif**. The preprocessed version of this dataset can be [downloaded](https://yadi.sk/d/5VdqLARizmnj3Q).
 [Check for details on this dataset](sup-mat/MGif/README.md). Training takes about  hours, on 1 gpu.

