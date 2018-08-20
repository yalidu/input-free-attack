# Region-Attack

This code is for our paper:

> Towards Query Efficient Black-box Attacks: An Input-free Perspective

## Setup

The code is tested under tensorflow-gpu=1.8 and python=3.6

To run the code, download [Inceptionv3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) classifier from Tensorflow website and put it in the directory `tools/data`.

## Run attacks

To run the region attack

```
python test_blackbox_attack.py  --momentum 0.7 --max-queries 1000000 --out-dir fast-query/ --save fast-attack-imgnet --size-selection -1
```

## Examples

### Attack [clarifai](https://clarifai.com/demo)

Please only select the food model. 

An example:

Target is Apple.

![apple](./examples/adv-img-Food-detection-clarifai-apple0.900.png "apple")
