# Input Free Attack - Region-Attack algorithm

This code is for our paper:

> Towards Query Efficient Black-box Attacks: An Input-free Perspective

## Setup

The code is tested under tensorflow-gpu=1.8 and python=3.6

To run the code, download [InceptionV3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) classifier from Tensorflow website and put it in the directory `tools/data`.

### Note

The infrastructure of the implementation for Natural Evolution Strategies is adapted from
[Black-box Adversarial Attacks with Limited Queries and Information](https://github.com/labsix/limited-blackbox-attacks).
Thank these contributors.

## Run attacks

To run the region attack

```
python test_blackbox_attack.py  --momentum 0.7 --max-queries 1000000 --out-dir fast-query/ --save fast-attack-imgnet --size-selection -1
```

## Examples

### Attack [clarifai](https://clarifai.com/demo)

Please only select the food model. An example is as follows:

Adversarial example| Category
-----------------------|-----------------------
![](./examples/adv-img-Food-detection-clarifai-apple0.900.png "apple") | apple

### Attack InceptionV3

Adversarial noise| Adversarial example| A ground-truth example| Category
-----------------------|-----------------------|-----------------------|-----------------------
![](./examples/146_diff_id1993.0_seq145_prev111_adv145_True_dist41.23348243836714.png)|![](./examples/146_adversarial_id1993.0_seq145_prev111_adv145_True_dist41.23348243836714.png)|![](./examples/146.00035982.jpg)| penguin
![](./examples/938_diff_id1993.0_seq937_prev111_adv937_True_dist31.206903987172115.png)|![](./examples/938_adversarial_id1993.0_seq937_prev111_adv937_True_dist31.206903987172115.png)|![](./examples/938.00043347.jpg)| broccoll
![](./examples/683_diff_id1993.0_seq682_prev111_adv682_True_dist37.59660364396521.png)|![](./examples/683_adversarial_id1993.0_seq682_prev111_adv682_True_dist37.59660364396521.png)|![](./examples/683.00034053.jpg)| obelisk
![](./examples/695_diff_id1993.0_seq694_prev111_adv694_True_dist49.44534568042813.png)|![](./examples/695_adversarial_id1993.0_seq694_prev111_adv694_True_dist49.44534568042813.png)|![](./examples/695.00038274.jpg)| paddle wheel

