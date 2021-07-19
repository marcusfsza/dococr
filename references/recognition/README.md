# Text recognition

The sample training script was made to train text recognition model with doctr

## Setup

First, you need to install doctr (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

if you are using PyTorch back-end, there is an extra dependency (to optimize data loading):
```shell
pip install contiguous-params>=1.0.0
```

## Usage

You can start your training in TensorFlow:

```shell
python references/recognition/train.py path/to/your/dataset crnn_vgg16_bn --epochs 5
```
or PyTorch:

```shell
python references/recognition/train_pytorch.py path/to/your/dataset crnn_vgg16_bn --epochs 5 --device 0
```



## Getting started

First, you need to install doctr (with pip, for instance)

```shell
pip install python-doctr
```

Then, to run the script execute the following command

```shell
python references/recognition/train.py crnn_vgg16_bn --epochs 5 --data_path path/to/your/dataset
```

## Data format

You need to provide a --data_path argument to start training. 
The data_path must lead to a 4-elements folder:

```shell
├── train
    ├── img_1.jpg
    ├── img_2.jpg
    ├── img_3.jpg
    └── ...
├── train_labels.json
├── val                    
    ├── img_a.jpg
    ├── img_b.jpg
    ├── img_c.jpg
    └── ...
├── val_labels.json
```

The JSON files must contain word-labels for each picture as a string. 
The order of entries in the json does not matter.

```shell
labels = {
    'img_1.jpg': 'I',
    'img_2.jpg': 'am',
    'img_3.jpg': 'a',
    'img_4.jpg': 'Jedi',
    'img_5.jpg': '!',
    ...
}
```

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/recognition/train.py --help
```
