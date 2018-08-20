import os
import shutil
import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Laplace
from tensorflow.contrib.distributions import Cauchy

distcCauchy = Cauchy(loc=0., scale=1.)
distLap = Laplace(loc=0., scale=1.)

NUM_LABELS = 1000
SIZE = 299
image_size = SIZE
inputs = np.zeros((SIZE, SIZE, 3)) + 0.5
num_channels = 3
orig_class = 111  # without backgound class its 111


class RegionAttack():
    def __init__(self, sess, args, model, gpus):
        self.sess = sess
        self.top_k = args.top_k
        self.sigma = args.sigma
        self.batch_size_main = args.batch_size
        self.probDistribution = args.probDistribution
        self.out_dir = args.out_dir
        self.epsilon = args.epsilon
        self.max_lr = args.max_lr
        self.min_lr = args.min_lr
        self.adv_thresh = args.adv_thresh
        self.starting_eps = args.starting_eps
        self.starting_delta_eps = args.starting_delta_eps
        self.momentum = args.momentum
        self.plateau_length = args.plateau_length
        self.plateau_drop = args.plateau_drop
        self.log_iters = args.log_iters
        self.learning_rate = args.learning_rate
        self.samples_per_draw = args.samples_per_draw
        self.gpus = gpus
        self.max_queries = args.max_queries

        print(">>search distribution", self.probDistribution)
        self.model = model
        self.labels = tf.placeholder(tf.int32, shape=[None, 1000])
        self.model_inputs = tf.placeholder(tf.float32, shape=[299, 299, 3])

        # ---region position---
        self.height = tf.Variable(32, np.int32)
        self.width = tf.Variable(32, np.int32)
        self.target_class = tf.Variable(1, np.int32)
        self.batch_size = tf.Variable(1, np.int32)

        self.assign_height = tf.placeholder(tf.int32)
        self.assign_width = tf.placeholder(tf.int32)
        self.assign_batch_size = tf.placeholder(tf.int32)
        self.assign_target_class = tf.placeholder(tf.int32)

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(tf.assign(self.height, self.assign_height))
        self.setup.append(tf.assign(self.width, self.assign_width))
        self.setup.append(
            tf.assign(self.target_class, self.assign_target_class))
        self.setup.append(tf.assign(self.batch_size, self.assign_batch_size))

        self.eval_logits, self.eval_preds = self.model(
            sess, tf.expand_dims(self.model_inputs, 0))
        self.eval_percent_adv = tf.equal(
            tf.cast(self.eval_preds[0], tf.int32), self.target_class)

        # GRADIENT ESTIMATION GRAPH
        self.loss_fn = self.standard_loss if self.top_k < 0 else self.partial_info_loss
        print(">>loss_fn", self.loss_fn)

        self.grad_estimates = []
        self.final_losses = []
        self.final_entropies = []
        pixshape = (self.height, self.width, num_channels)
        batch_per_gpu = self.batch_size // len(gpus)
        for i, device in enumerate(gpus):
            with tf.device(device):
                if self.probDistribution == "gauss":
                    # gauss
                    noise_pos = tf.random_normal((batch_per_gpu // 2, ) +
                                                 pixshape)
                elif self.probDistribution == "cauchy":
                    # cauchy
                    noise_pos = distcCauchy.sample(
                        (batch_per_gpu // 2, ) +
                        pixshape)  # cauchy distribution
                elif self.probDistribution == "laplace":
                    # laplace
                    noise_pos = distLap.sample((batch_per_gpu // 2, ) +
                                               pixshape)
                else:
                    raise ValueError(
                        "The distribution is not considered. Please compose the code to implement it"
                    )

                hno, wno = tf.div(image_size, self.height), tf.div(
                    image_size, self.width)
                hr, wr = tf.mod(image_size, self.height), tf.mod(
                    image_size, self.width)

                tmp = tf.tile(noise_pos, [1, hno, wno, 1])
                noise_pos = tf.pad(tmp, [[0, 0], [hr // 2, hr - hr // 2],
                                         [wr // 2, wr - wr // 2], [0, 0]],
                                   "SYMMETRIC")
                noise = tf.concat([noise_pos, -noise_pos], axis=0)
                eval_points = self.model_inputs + self.sigma * noise

                losses, noise = self.loss_fn(eval_points, noise)
                entropies = self.entropy_loss(eval_points)
            losses_tiled = tf.tile(
                tf.reshape(losses, (-1, 1, 1, 1)), (1, ) + inputs.shape)

            if self.probDistribution == "gauss":
                # gauss
                self.grad_estimates.append(
                    tf.reduce_mean(losses_tiled * noise, axis=0) / self.sigma)
            elif self.probDistribution == "cauchy":
                # cauchy
                self.grad_estimates.append(
                    tf.reduce_mean(
                        losses_tiled * noise / (noise * noise + 1), axis=0) * 2
                    / self.sigma)
            elif self.probDistribution == "laplace":
                # laplace
                self.grad_estimates.append(
                    tf.reduce_mean(losses_tiled * tf.sign(noise), axis=0) /
                    self.sigma)
            else:
                raise ValueError(
                    "The distribution is not considered. Please compose the code to implement its derivative"
                )

            self.final_losses.append(losses)
            self.final_entropies.append(entropies)

        self.grad_estimate = tf.reduce_mean(self.grad_estimates, axis=0)
        self.final_losses = tf.concat(self.final_losses, axis=0)
        self.final_entropy = tf.concat(self.final_entropies, axis=0)

    def predict(self, img):
        eval_logits_, eval_preds_ = self.sess.run(
            [self.eval_logits, self.eval_preds],
            feed_dict={self.model_inputs: img})
        return eval_logits_, eval_preds_

    def standard_loss(self, eval_points, noise):
        logits, preds = self.model(self.sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels)
        return losses, noise

    def partial_info_loss(self, eval_points, noise):
        logits, preds = self.model(self.sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels)
        vals, inds = tf.nn.top_k(logits, k=self.top_k)
        # inds is batch_size x k
        good_inds = tf.where(tf.equal(
            inds, self.target_class))  # returns (# true) x 3
        good_images = good_inds[:, 0]  # inds of img in batch that worked
        losses = tf.gather(losses, good_images)
        noise = tf.gather(noise, good_images)
        return losses, noise

    # entropy loss
    def entropy_loss(self, eval_points):
        logits, preds = self.model(self.sess, eval_points)
        probs = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=probs)
        return entropy

    # GRADIENT ESTIMATION EVAL
    def get_grad(self, img, label_gt, spd, bs):
        num_batches = spd // bs
        losses = []
        grads = []
        l_entropies = []

        feed_dict = {self.model_inputs: img, self.labels: label_gt}

        for _ in range(num_batches):
            loss, dl_dx_, loss_entropy = self.sess.run(
                [self.final_losses, self.grad_estimate, self.final_entropy],
                feed_dict=feed_dict)
            losses.append(np.mean(loss))
            grads.append(dl_dx_)
            l_entropies.append(loss_entropy)

        return np.array(losses).mean(), np.mean(
            np.array(grads), axis=0), np.array(l_entropies).mean()

    def robust_in_top_k(self, t_, prop_adv_, k_):
        if self.top_k == NUM_LABELS:
            return True
        for i in range(1):
            n = np.random.rand(*prop_adv_.shape) * self.sigma
            eval_logits_ = self.sess.run(
                self.eval_logits, feed_dict={self.model_inputs: prop_adv_})[0]
            if not t_ in eval_logits_.argsort()[-k_:][::-1]:
                return False
        return True

    def warm_up(self, initial_img, opt):
        sess = self.sess
        gpus = self.gpus
        try:
            height = opt['height']
            width = opt['width']
        except:
            height, width = 64, 64
        target_class_real = opt['target']
        batch_size_warm = 4
        out_dir = self.out_dir
        epsilon = self.epsilon
        sess.run(
            self.setup, {
                self.assign_height: height,
                self.assign_width: width,
                self.assign_target_class: target_class_real,
                self.assign_batch_size: batch_size_warm
            })

        conservative = 5
        lower = np.clip(initial_img - 1, 0., 1.)
        upper = np.clip(initial_img + 1, 0., 1.)
        #adv = initial_img.copy() if not args.restore else \
        #    np.clip(np.load(args.restore), lower, upper)
        adv = initial_img.copy()
        batch_per_gpu = batch_size_warm // len(gpus)

        queries_per_iter = batch_size_warm
        max_iters = 5  # for select region
        print("(warmup) batch_size:", batch_size_warm, "max_iters", max_iters)
        max_lr = self.max_lr
        min_lr = self.min_lr
        # ----- partial info params -----
        k = self.top_k
        goal_epsilon = epsilon
        adv_thresh = self.adv_thresh
        if k > 0:
            epsilon = self.starting_eps
            delta_epsilon = self.starting_delta_eps
        else:
            k = NUM_LABELS
        # ----- label only params -----
        #label_only = args.label_only
        #zero_iters = args.zero_iters

        one_hot_vec = self.one_hot(target_class_real, NUM_LABELS)
        labels = np.repeat(
            np.expand_dims(one_hot_vec, axis=0), repeats=batch_per_gpu, axis=0)
        is_targeted = 1

        # HISTORY VARIABLES (for backtracking and momentum)
        num_queries = 0
        g = 0
        prev_adv = adv
        last_ls = []

        # MAIN LOOP
        lall = []
        gall = []
        eall = []
        for i in range(max_iters):
            start = time.time()

            # CHECK IF WE SHOULD STOP
            padv = sess.run(
                [self.eval_percent_adv], feed_dict={self.model_inputs: adv})

            if padv == 1 and epsilon <= goal_epsilon:
                print('[log] early stopping at iteration %d' % i)
                break

            prev_g = g
            l, g, l_ent = self.get_grad(adv, labels, queries_per_iter,
                                        batch_size_warm)

            lall.append(l)
            gall.append(g)
            eall.append(l_ent)

            # SIMPLE MOMENTUM
            g = self.momentum * prev_g + (1.0 - self.momentum) * g

            # PLATEAU LR ANNEALING
            plateau_length = self.plateau_length
            plateau_drop = self.plateau_drop
            last_ls.append(l)
            last_ls = last_ls[-plateau_length:]
            if last_ls[-1] > last_ls[0] \
                    and len(last_ls) == plateau_length:
                if max_lr > min_lr:
                    max_lr = max(max_lr / plateau_drop, min_lr)
                    print("[log] Annealing max_lr=", max_lr)
                last_ls = []

            # SEARCH FOR LR AND EPSILON DECAY
            current_lr = max_lr
            # proposed_adv = adv - is_targeted * current_lr * np.sign(g)
            prop_de = 0.0
            if l < adv_thresh and epsilon > goal_epsilon:
                prop_de = delta_epsilon
            while current_lr >= min_lr:
                # GENERAL LINE SEARCH
                proposed_adv = adv - is_targeted * current_lr * np.sign(g)

                proposed_adv = np.clip(proposed_adv, lower, upper)
                num_queries += 1
                if self.robust_in_top_k(target_class_real, proposed_adv, k):
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, 0.1)
                        last_ls = []
                    prev_adv = adv
                    adv = proposed_adv
                    epsilon = max(epsilon - prop_de / conservative,
                                  goal_epsilon)
                    break
                elif current_lr >= min_lr * 2:
                    current_lr = current_lr / 2
                else:
                    prop_de = prop_de / 2
                    if prop_de == 0:
                        raise ValueError("Did not converge.")
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    print("[log] backtracking eps to %3f" %
                          (epsilon - prop_de, ))

            # BOOK-KEEPING STUFF
            num_queries += queries_per_iter
            if i % 4 == 0:
                log_text = 'Step {:d}, loss={:.4f}, lr={:.2E}, eps={:.3f}(time {:.4f}), size=({},{})'.format(
                    i, l, current_lr, epsilon,
                    time.time() - start, sess.run(self.height),
                    sess.run(self.width))
                print(log_text)

                sys.stdout.flush()

        return adv, num_queries, lall, gall, eall

    def one_hot(self, index, total):
        arr = np.zeros((total))
        arr[index] = 1.0
        return arr

    def attack_fast(self, initial_img, opt):
        sess = self.sess
        gpus = self.gpus
        try:
            height = opt['height']
            width = opt['width']
        except:
            height, width = 64, 64

        target_class_real = opt['target']

        print("batch_size:", self.batch_size_main)

        batch_size_attack = self.batch_size_main
        epsilon = self.epsilon

        sess.run(
            self.setup, {
                self.assign_height: height,
                self.assign_width: width,
                self.assign_target_class: target_class_real,
                self.assign_batch_size: self.batch_size_main
            })

        conservative = 10
        lower = np.clip(initial_img - 1, 0., 1.)
        upper = np.clip(initial_img + 1, 0., 1.)
        #adv = initial_img.copy() if not args.restore else \
        #    np.clip(np.load(args.restore), lower, upper)
        adv = initial_img.copy()
        batch_per_gpu = batch_size_attack // len(gpus)
        log_iters = self.log_iters
        current_lr = self.learning_rate
        queries_per_iter = self.samples_per_draw
        max_iters = int(np.ceil(self.max_queries // queries_per_iter))
        max_lr = self.max_lr
        # ----- partial info params -----
        k = self.top_k
        goal_epsilon = epsilon
        adv_thresh = self.adv_thresh
        if k > 0:
            if target_class_real == -1:
                raise ValueError(
                    "Partial-information attack is a targeted attack.")
            epsilon = self.starting_eps
            delta_epsilon = self.starting_delta_eps
        else:
            k = NUM_LABELS

        # TARGET CLASS SELECTION
        if target_class_real < 0:
            one_hot_vec = self.one_hot(orig_class, NUM_LABELS)
        else:
            one_hot_vec = self.one_hot(target_class_real, NUM_LABELS)

        labels = np.repeat(
            np.expand_dims(one_hot_vec, axis=0), repeats=batch_per_gpu, axis=0)
        is_targeted = 1 if target_class_real >= 0 else -1

        # HISTORY VARIABLES (for backtracking and momentum)
        num_queries = 0
        g = 0
        prev_adv = adv
        last_ls = []

        # MAIN LOOP
        for i in range(max_iters):
            start = time.time()

            # CHECK IF WE SHOULD STOP
            padv = sess.run(
                self.eval_percent_adv, feed_dict={self.model_inputs: adv})
            if padv == 1 and epsilon <= goal_epsilon:
                print('[log] early stopping at iteration %d' % i)
                break

            prev_g = g
            l, g, _ = self.get_grad(adv, labels, self.samples_per_draw,
                                    batch_size_attack)

            # SIMPLE MOMENTUM
            g = self.momentum * prev_g + (1.0 - self.momentum) * g

            # PLATEAU LR ANNEALING
            last_ls.append(l)
            last_ls = last_ls[-self.plateau_length:]
            if last_ls[-1] > last_ls[0] \
                    and len(last_ls) == self.plateau_length:
                if max_lr > self.min_lr:
                    max_lr = max(max_lr / self.plateau_drop, self.min_lr)
                    print("[log] Annealing max_lr=", max_lr)

                last_ls = []

            # SEARCH FOR LR AND EPSILON DECAY
            current_lr = max_lr
            # proposed_adv = adv - is_targeted * current_lr * np.sign(g)
            prop_de = 0.0
            if l < adv_thresh and epsilon > goal_epsilon:
                prop_de = delta_epsilon
            while current_lr >= self.min_lr:
                proposed_adv = adv - is_targeted * current_lr * np.sign(g)

                proposed_adv = np.clip(proposed_adv, lower, upper)
                num_queries += 1
                if self.robust_in_top_k(target_class_real, proposed_adv, k):
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, 0.1)
                        last_ls = []
                    prev_adv = adv
                    adv = proposed_adv
                    epsilon = max(epsilon - prop_de / conservative,
                                  goal_epsilon)
                    break
                elif current_lr >= self.min_lr * 2:
                    current_lr = current_lr / 2
                    # print("[log] backtracking lr to %3f" % (current_lr,))
                else:
                    prop_de = prop_de / 2
                    if prop_de == 0:
                        break
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    print("[log] backtracking eps to %3f" %
                          (epsilon - prop_de, ))

            # BOOK-KEEPING STUFF
            num_queries += self.samples_per_draw
            if i % 20 == 0:
                log_text = 'Step {:03d}, loss={:.4f}, lr={:.2E}, eps={:.3f}(time {:.4f}), size=({},{})'.format(
                    i, l, current_lr, epsilon,
                    time.time() - start, sess.run(self.height),
                    sess.run(self.width))
                print(log_text)
                sys.stdout.flush()

        return adv, num_queries

    def active_select_size(self, inputs, candiates, targets):
        #candiates.sort()
        ls = []
        gs = []
        es = []  # entropy of iterations
        advs = []
        for kk in range(len(candiates)):
            opt = {}
            opt['height'] = candiates[kk]
            opt['width'] = candiates[kk]
            opt['target'] = np.argmax(targets)
            adv, eval_costs, l, g, l_ent = self.warm_up(inputs, opt)
            ls.append(l)
            gs.append(g)
            es.append(l_ent)
            advs.append(adv)

        best_ind = self.get_size_min_ent(es)
        fh = candiates[best_ind]
        fw = candiates[best_ind]
        adv_img = advs[best_ind]
        print(">>Candiates:", candiates)
        print(">>Selected height={}, width={}".format(fh, fw))
        return fh, fw, adv_img

    def get_size_min_ent(self, es):
        large_ent = 0.
        # small_ent = 100.
        selected = 0
        for i in range(len(es)):
            tmp = es[i]
            min_v = min(tmp)
            min_ind = tmp.index(min_v)
            after_min = tmp[min_ind + 1:]
            if after_min == []:
                delta_ent = 0.
            else:
                ent_t = es[i][len(es[i]) - 1]
                delta_ent = ent_t - min_v
            if delta_ent > large_ent:
                large_ent = delta_ent
                selected = i
        return selected
