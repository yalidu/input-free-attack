import os
import json
import shutil
import argparse
import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pdb
from tools.inception_v3_imagenet import model
import tensorflow as tf
import random
import helper
import scipy
from PIL import Image

from fast_attack import RegionAttack

random.seed(12161)
np.random.seed(12161)
tf.set_random_seed(12161)

SIGMA = 1e-3
EPSILON = 0.05
SAMPLES_PER_DRAW = 20
BATCH_SIZE = SAMPLES_PER_DRAW
LEARNING_RATE = 1e-2
#LOG_ITERS_FACTOR = 2
SIZE = 299
orig_class = 111  # without backgound class its 111


def parse_args():
    ## parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--samples-per-draw', type=int, default=SAMPLES_PER_DRAW)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument(
        '--target-class', type=int, help='negative => untargeted')
    parser.add_argument('--orig-class', type=int)
    parser.add_argument('--sigma', type=float, default=SIGMA)
    parser.add_argument('--epsilon', type=float, default=EPSILON)
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--img-index', type=int, default=0)
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='dir to save to if not gridding; otherwise parent \
                                  dir of grid directories')
    parser.add_argument('--log-iters', type=int, default=1)
    parser.add_argument('--restore', type=str, help='restore path of img')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--save-iters', type=int, default=50)
    parser.add_argument('--plateau-drop', type=float, default=2.0)
    parser.add_argument('--min-lr-ratio', type=int, default=200)
    parser.add_argument('--plateau-length', type=int, default=5)
    parser.add_argument('--gpus', type=int, help='number of GPUs to use')
    parser.add_argument('--imagenet-path', type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max-lr', type=float, default=1e-2)
    parser.add_argument('--min-lr', type=float, default=5e-5)
    # PARTIAL INFORMATION ARGUMENTS
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--adv-thresh', type=float, default=-1.0)
    # LABEL ONLY ARGUMENTS
    parser.add_argument('--label-only', action='store_true')
    parser.add_argument(
        '--zero-iters',
        type=int,
        default=100,
        help="how many points to use for the proxy score")
    parser.add_argument(
        '--label-only-sigma',
        type=float,
        default=1e-3,
        help="distribution width for proxy score")
    parser.add_argument('--starting-eps', type=float, default=1.0)
    parser.add_argument('--starting-delta-eps', type=float, default=0.5)
    parser.add_argument('--min-delta-eps', type=float, default=0.1)
    parser.add_argument(
        '--conservative',
        type=int,
        default=2,
        help=
        "How conservative we should be in epsilon decay; increase if no convergence"
    )
    parser.add_argument('--save', type=str, help="save results")
    parser.add_argument("--seed", type=int, default=12163)
    parser.add_argument("--size-selection", type=int, default=-1)
    parser.add_argument("--batch-size-warmup", type=int, default=-1)
    parser.add_argument(
        "--probDistribution",
        choices=["gauss", "cauchy", "laplace"],
        default="gauss")
    return parser.parse_args()


def main(args):
    # PRINT PARAMS
    args_text = json.dumps(args.__dict__)
    print(args_text)
    # data checks
    if not (args.img_path is None and args.img_index is not None
            or args.img_path is not None and args.img_index is None):
        raise ValueError('can only set one of img-path, img-index')
    if args.img_path and not (args.orig_class or args.target_class):
        raise ValueError('orig and target class required with image path')
    if (args.target_class is None and args.img_index is None):
        raise ValueError('must give target class if not using index')

    assert args.samples_per_draw % args.batch_size == 0

    gpus = helper.get_available_gpus()
    print("available_gpus:", gpus)
    if args.gpus:
        if args.gpus > len(gpus):
            raise RuntimeError('not enough GPUs! (requested %d, found %d)' %
                               (args.gpus, len(gpus)))
        gpus = gpus[:args.gpus]
    if not gpus:
        raise NotImplementedError('no support for using CPU-only because lazy')
    if args.batch_size % (2 * len(gpus)) != 0:
        raise ValueError(
            'batch size must be divisible by 2 * number of GPUs (batch_size=%d, gpus=%d)'
            % (BATCH_SIZE, len(gpus)))
    print("gpus:", gpus)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    attack = RegionAttack(sess, args, model, gpus)

    # an input gray image
    num_targets = 1000
    SIZE = 299
    initial_img = np.zeros((SIZE, SIZE, 3)) + 0.5

    dir_prefix = args.out_dir
    os.system("mkdir -p {}/{}".format(dir_prefix, args.save))

    filename = "{}/{}.txt".format(dir_prefix, args.save)
    myfile = open(filename, 'a+')
    myfile.write(args_text + '\n')
    myfile.close()

    # main loop
    # generate data
    # generate gray imgs for attacking
    all_targets = np.eye(num_targets)
    all_inputs = [initial_img for i in range(num_targets)]
    all_inputs = np.asarray(all_inputs)

    original_predict, _ = attack.predict(initial_img)
    original_predict = np.squeeze(original_predict)
    original_class = np.argsort(original_predict)

    all_labels = np.zeros((num_targets, num_targets))
    all_labels[:, original_class[num_targets -
                                 1]] = 1  # true labels for gray img

    print('Done of generating data')
    img_no = 0
    total_success = 0
    l2_total = 0.0
    eval_cost_total = 0

    start_ID = 0
    #test_classes = num_targets
    test_classes = 5  # for debug purpose

    for i in range(start_ID, test_classes):
        inputs = np.squeeze(all_inputs[i:i + 1])
        targets = all_targets[i:i + 1]
        labels_real = all_labels[i:i + 1]
        print("true label:", np.argmax(labels_real))
        print("target:", np.argmax(targets))
        if np.argmax(targets) == orig_class:
            print("**Original class", orig_class)
            continue
        # test if the image is correctly classified
        oritinal_predict, _ = attack.predict(initial_img)
        original_predict = np.squeeze(original_predict)
        original_prob = np.sort(original_predict)
        original_class = np.argsort(original_predict)

        print("original probabilities:", original_prob[-1:-6:-1])
        print("original classification:", original_class[-1:-6:-1])
        print("original probabilities (most unlikely):", original_prob[:6])
        print("original classification (most unlikely):", original_class[:6])

        if args.size_selection > 0:
            # if True, we perform size selection
            # size_candidates = np.random.choice(range(32, 150), 10, replace=False)
            size_candidates = np.random.choice(
                range(32, 150), 2, replace=False)
            fh, fw, adv_init = attack.active_select_size(
                inputs, size_candidates, targets)
        else:
            # set a default size which works well
            fw = 64
            adv_init = initial_img

        regs = np.array([fw])
        for k_size in regs:
            fh, fw = k_size, k_size
            opt = {}
            opt['height'] = fh
            opt['width'] = fw
            opt['target'] = np.argmax(targets)

            img_no += 1

            timestart = time.time()
            if np.argmax(targets) == np.argmax(original_predict):
                adv, eval_costs = inputs, 1
            else:
                adv, eval_costs = attack.attack_fast(adv_init, opt)

            timeend = time.time()
            l2_distortion = np.sum((adv - inputs)**2)**.5
            l0_distortion = np.sum(abs(adv - inputs) > 0)
            l1_distortion = np.sum(abs(adv - inputs))
            li_distortion = np.max(abs(adv - inputs))

            adversarial_predict, _ = attack.predict(adv)

            adversarial_predict = np.squeeze(adversarial_predict)
            adversarial_prob = np.sort(adversarial_predict)
            adversarial_class = np.argsort(adversarial_predict)
            print("adversarial probabilities:", adversarial_prob[-1:-6:-1])
            print("adversarial classification:", adversarial_class[-1:-6:-1])
            success = False

            if adversarial_class[-1] == np.argmax(targets):
                success = True
            if success:
                total_success += 1
                l2_total += l2_distortion
                eval_cost_total += eval_costs
            suffix = "id{}_seq{}_prev{}_adv{}_{}_dist{}".format(
                img_no, i, original_class[-1], adversarial_class[-1], success,
                l2_distortion)
            print("Saving to", suffix)
            helper.show(
                adv, "{}/{}/{}_adversarial_{}.png".format(
                    dir_prefix, args.save, img_no, suffix))
            helper.show(
                adv - inputs, "{}/{}/{}_diff_{}.png".format(
                    dir_prefix, args.save, img_no, suffix))

            print(
                "[STATS][L1] total={}, seq={}, time={:.3f}, prev_class={}, new_class={}, distortion={:.5f}, l2_avg={:.5f}, success_rate={:.6f}, query={}, avg_query={:.1f},  !!!success!!!={}\n".
                format(
                    img_no, i, timeend - timestart, original_class[-1],
                    adversarial_class[-1], l2_distortion, 0
                    if total_success == 0 else l2_total / total_success,
                    total_success / float(img_no), eval_costs, 0
                    if total_success == 0 else eval_cost_total / total_success,
                    success))

            sys.stdout.flush()
            str1 = "img_no:{} prev:{} adv_label:{} Success:{} time:{:.3f} eval_costs:{} l2_dist:{:.3f} l0_dist:{} l1_dist:{} li_dist:{:.6f} success_rate:{:.5f} avg_query:{:.3f} size:{}\n".format(
                img_no, original_class[-1], adversarial_class[-1], success,
                timeend - timestart, eval_costs, l2_distortion, l0_distortion,
                l1_distortion, li_distortion, total_success / float(img_no), 0
                if total_success == 0 else eval_cost_total / total_success, fh)
            myfile = open(filename, 'a+')
            myfile.write(str1)
            myfile.close()
    success_rate = total_success / img_no
    l2_distortion_average = 0 if total_success == 0 else l2_total / total_success
    eval_cost_average = eval_cost_total / total_success
    myfile = open(filename, 'a+')
    myfile.write("success_rate:%f average_cost:%d average_l2_distortion:%.3f\n"
                 % (success_rate, eval_cost_average, l2_distortion_average))
    myfile.close()
    # close the file
    print("success_rate:%.3f average_cost:%d average_l2_distortion:%.3f" %
          (success_rate, eval_cost_average, l2_distortion_average))


if __name__ == "__main__":
    args = parse_args()
    main(args)
