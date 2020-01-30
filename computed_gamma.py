# This code is a revisitation of the code by 

# Get and unpack data

#!mkdir data

#!mkdir ./data/cifar10
#!wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#!tar -xvzf cifar-10-python.tar.gz -C ./data/cifar10

#!mkdir ./data/celeba
#!kaggle datasets download -d jessicali9530/celeba-dataset
#!unzip celeba-dataset.zip
#!unzip -q img_align_celeba.zip
#!mv list_eval_partition.csv list_landmarks_align_celeba.csv list_bbox_celeba.csv list_attr_celeba.csv ./data/celeba
#!mv img_align_celeba ./data/celeba/img_align_celeba

### Preprocess data

import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
from imageio import imread, imwrite

ROOT_FOLDER = './data'

def load_cifar10_data(flag='training'):
    if flag == 'training':
        data_files = ['data/cifar10/cifar-10-batches-py/data_batch_1', 'data/cifar10/cifar-10-batches-py/data_batch_2', 'data/cifar10/cifar-10-batches-py/data_batch_3', 'data/cifar10/cifar-10-batches-py/data_batch_4', 'data/cifar10/cifar-10-batches-py/data_batch_5']
    else:
        data_files = ['data/cifar10/cifar-10-batches-py/test_batch']
    x = []
    for filename in data_files:
        img_dict = unpickle(filename)
        img_data = img_dict[b'data']
        img_data = np.transpose(np.reshape(img_data, [-1, 3, 32, 32]), [0, 2, 3, 1])
        x.append(img_data)
    x = np.concatenate(x, 0)
    num_imgs = np.shape(x)[0]
    
    # save to jpg file
    img_folder = os.path.join('data/cifar10', flag)
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    for i in range(num_imgs):
        imwrite(os.path.join(img_folder, str(i) + '.jpg'), x[i])

    # save to npy
    x = []
    for i in range(num_imgs):
        img_file = os.path.join(img_folder, str(i) + '.jpg')
        img = imread(img_file, pilmode='RGB')
        x.append(np.reshape(img, [1, 32, 32, 3]))
    x = np.concatenate(x, 0)

    return x.astype(np.uint8)


def load_celeba_data(flag='training', side_length=None, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[45:173,25:153]
        img = np.array(Image.fromarray(img).resize((side_length, side_length),resample=Image.BILINEAR))
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)

# Center crop 128x128 and resize to 64x64
def preprocess_celeba():
    x_val = load_celeba_data('val', 64)
    np.save(os.path.join('data', 'celeba', 'val.npy'), x_val)
    x_test = load_celeba_data('test', 64)
    np.save(os.path.join('data', 'celeba', 'test.npy'), x_test)
    x_train = load_celeba_data('training', 64)
    np.save(os.path.join('data', 'celeba', 'train.npy'), x_train)

def preporcess_cifar10():
    x_train = load_cifar10_data('training')
    np.save(os.path.join('data', 'cifar10', 'train.npy'), x_train)
    x_test = load_cifar10_data('testing')
    np.save(os.path.join('data', 'cifar10', 'test.npy'), x_test)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo) #, encoding='bytes')
    return dic

#uncomment this line to create the datatset
#preprocess_celeba()
#preporcess_cifar10()

def load_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels

def load_test_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels

#######################################################################
import tensorflow as tf 
from tensorflow.contrib import layers 
import math 
import numpy as np 
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           initializer=tf.truncated_normal_initializer):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
    biases = tf.get_variable(
        "biases", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

def lrelu(input_, leak=0.2, name="lrelu"):
      return tf.maximum(input_, leak * input_, name=name)

def batch_norm(x, is_training, scope, eps=1e-5, decay=0.999, affine=True):
    def mean_var_with_update(moving_mean, moving_variance):
        if len(x.get_shape().as_list()) == 4:
            statistics_axis = [0, 1, 2]
        else:
            statistics_axis = [0]
        mean, variance = tf.nn.moments(x, statistics_axis, name='moments')
        with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay), assign_moving_average(moving_variance, variance, decay)]):
            return tf.identity(mean), tf.identity(variance)

    with tf.name_scope(scope):
        with tf.variable_scope(scope + '_w'):
            params_shape = x.get_shape().as_list()[-1:]
            moving_mean = tf.get_variable('mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = tf.get_variable('variance', params_shape, initializer=tf.ones_initializer, trainable=False)

            mean, variance = tf.cond(is_training, lambda: mean_var_with_update(moving_mean, moving_variance), lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                return tf.nn.batch_normalization(x, mean, variance, None, None, eps)


def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable("biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

def downsample(x, out_dim, kernel_size, l2_reg, name):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert(len(input_shape) == 4)
        #return tf.layers.conv2d(x, out_dim, kernel_size, 2, 'same')
        return (tf.keras.layers.Conv2D(out_dim, kernel_size, strides=2, padding='same',kernel_regularizer=regularizers.l2(l2_reg))(x))

def upsample(x, out_dim, kernel_size, l2_reg, name):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()
        assert(len(input_shape) == 4)
        #return tf.layers.conv2d_transpose(x, out_dim, kernel_size, 2, 'same')
        #return (tf.keras.layers.Conv2DTranspose(out_dim, kernel_size, strides=2, padding='same',kernel_regularizer='l2',activity_regularizer='l2')(x))
        return (tf.keras.layers.Conv2DTranspose(out_dim, kernel_size, strides=2, padding='same',kernel_regularizer=regularizers.l2(l2_reg))(x))


def res_block(x, out_dim, is_training, name, depth=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x
        for i in range(depth):
            y = tf.nn.relu(batch_norm(y, is_training, 'bn'+str(i)))
            y = tf.layers.conv2d(y, out_dim, kernel_size, padding='same', name='layer'+str(i))
        s = tf.layers.conv2d(x, out_dim, kernel_size, padding='same', name='shortcut')
        return y + s 


def res_fc_block(x, out_dim, name, depth=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(depth):
            y = tf.layers.dense(tf.nn.relu(y), out_dim, name='layer'+str(i))
        s = tf.layers.dense(x, out_dim, name='shortcut')
        return y + s 


def scale_block(x, out_dim, is_training, name, block_per_scale=1, depth_per_block=2, kernel_size=3):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_block(y, out_dim, is_training, 'block'+str(i), depth_per_block, kernel_size)
        return y 


def scale_fc_block(x, out_dim, name, block_per_scale=1, depth_per_block=2):
    with tf.variable_scope(name):
        y = x 
        for i in range(block_per_scale):
            y = res_fc_block(y, out_dim, 'block'+str(i), depth_per_block)
        return y 


### Model

import tensorflow as tf 
import math 
import numpy as np 
from tensorflow.python.training.moving_averages import assign_moving_average


class TwoStageVaeModel(object):
    def __init__(self, x, latent_dim=64, second_depth=3, second_dim=1024):
        self.raw_x = x
        self.x = tf.cast(self.raw_x, tf.float32) / 255.0 
        self.batch_size = x.get_shape().as_list()[0]
        self.latent_dim = latent_dim
        self.second_dim = second_dim
        self.img_dim = x.get_shape().as_list()[1]

        self.second_depth = second_depth

        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        self.gamma_x = tf.placeholder(tf.float32, [], 'gamma_x')
        self.gamma_z = tf.placeholder(tf.float32, [], 'gamma_z')

        self.__build_network()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()

    def __build_network(self):
        with tf.variable_scope('stage1'):
            self.build_encoder1()
            self.build_decoder1()
        with tf.variable_scope('stage2'):
            self.build_encoder2()
            self.build_decoder2()

    def __build_loss(self):
        HALF_LOG_TWO_PI = 0.91893
        k = (2*self.img_dim/self.latent_dim)**2
        self.kl_loss1 = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size)
        self.loggamma_x = tf.log(self.gamma_x)
        self.gen_loss1 = tf.reduce_sum(tf.square((self.x - self.x_hat) / self.gamma_x) / 2.0 + self.loggamma_x + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss1 = k*self.kl_loss1 + self.gen_loss1 

        self.loggamma_z = tf.log(self.gamma_z)
        self.kl_loss2 = tf.reduce_sum(tf.square(self.mu_u) + tf.square(self.sd_u) - 2 * self.logsd_u - 1) / 2.0 / float(self.batch_size)
        self.mse_loss2 = tf.losses.mean_squared_error(self.z, self.z_hat)
        self.gen_loss2 = tf.reduce_sum(tf.square((self.z - self.z_hat) / self.gamma_z) / 2.0 + self.loggamma_z + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss2 = self.kl_loss2 + self.gen_loss2 

    def __build_summary(self):
        with tf.name_scope('stage1_summary'):
            self.summary1 = []
            self.summary1.append(tf.summary.scalar('gamma', self.gamma_x))
            self.summary1 = tf.summary.merge(self.summary1)

        with tf.name_scope('stage2_summary'):
            self.summary2 = []
            self.summary2.append(tf.summary.scalar('gamma', self.gamma_z))
            self.summary2 = tf.summary.merge(self.summary2)

    def __build_optimizer(self):
        all_variables = tf.global_variables()
        variables1 = [var for var in all_variables if 'stage1' in var.name]
        variables2 = [var for var in all_variables if 'stage2' in var.name]
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
        self.opt1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss1, self.global_step, var_list=variables1)
        self.opt2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss2, self.global_step, var_list=variables2)
        
    def build_encoder2(self):
        with tf.variable_scope('encoder'):
            t = self.z 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.z, t], -1)
        
            self.mu_u = tf.layers.dense(t, self.latent_dim, name='mu_u')
            self.logsd_u = tf.layers.dense(t, self.latent_dim, name='logsd_u')
            self.sd_u = tf.exp(self.logsd_u)
            self.u = self.mu_u + self.sd_u * tf.random_normal([self.batch_size, self.latent_dim])
        
    def build_decoder2(self):
        with tf.variable_scope('decoder'):
            t = self.u 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.u, t], -1)
            self.z_hat = tf.layers.dense(t, self.latent_dim, name='z_hat')

    def extract_posterior(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        mu_z, sd_z = [], []
        for i in range(num_iter):
            mu_z_batch, sd_z_batch = sess.run([self.mu_z, self.sd_z], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            mu_z.append(mu_z_batch)
            sd_z.append(sd_z_batch)
        mu_z = np.concatenate(mu_z, 0)[0:num_sample]
        sd_z = np.concatenate(sd_z, 0)[0:num_sample]
        return mu_z, sd_z

    def step(self, stage, input_batch, gamma, lr, sess, writer=None, write_iteration=600):
        if stage == 1:
            loss, summary, mse_loss,_ = sess.run([self.loss1, self.summary1, self.mse_loss1, self.opt1], feed_dict={self.raw_x: input_batch, self.gamma_x: gamma, self.lr: lr, self.is_training: True})
        elif stage == 2:
            loss, summary, mse_loss,_ = sess.run([self.loss2, self.summary2, self.mse_loss2, self.opt2], feed_dict={self.z: input_batch, self.gamma_z:gamma,self.lr: lr, self.is_training: True})
        else:
            raise Exception('Wrong stage {}.'.format(stage))
        global_step = self.global_step.eval(sess)
        if global_step % write_iteration == 0 and writer is not None:
            writer.add_summary(summary, global_step)
        return loss, mse_loss

    def reconstruct2(self, sess, z):
        #reconstruction of latent space by the second stage
        num_sample = np.shape(z)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        z_extend = np.concatenate([z, z[0:self.batch_size]], 0)
        recon_z = []
        for i in range(num_iter):
            recon_z_batch = sess.run(self.z_hat, feed_dict={self.z: z_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            recon_z.append(recon_z_batch)
        recon_z = np.concatenate(recon_z, 0)[0:num_sample]
        return recon_z
    
    def reconstruct(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        recon_x = []
        mu_z_tot = []
        logsd_z_tot = []
        for i in range(num_iter):
            #get mu_z and logsd_z for every batch of data
            mu_z_batch, logsd_z_batch, recon_x_batch = sess.run([self.mu_z, self.logsd_z, self.x_hat], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False}) 
            recon_x.append(recon_x_batch)
            mu_z_tot.append(mu_z_batch)
            logsd_z_tot.append(logsd_z_batch)
        recon_x = np.concatenate(recon_x, 0)[0:num_sample]
        mu_z_tot = np.concatenate(mu_z_tot, 0)[0:num_sample]
        logsd_z_tot = np.concatenate(logsd_z_tot, 0)[0:num_sample]
        #return recon_x
        return mu_z_tot, logsd_z_tot, recon_x

    def generate(self, sess, num_sample, stage=2):
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        gen_samples = []
        gen_z = []
        for i in range(num_iter):
            if stage == 2:
                # u ~ N(0, I)
                u = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
                # z ~ N(f_2(u), \gamma_z I)
                z = sess.run(self.z_hat, feed_dict={self.u: u, self.is_training: False})
                z = z/np.mean(np.std(z,axis=0))
            else:
                z = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            # x = f_1(z)
            x = sess.run(self.x_hat, feed_dict={self.z: z, self.is_training: False})
            gen_z.append(z)
            gen_samples.append(x)
        gen_z = np.concatenate(gen_z, 0)
        gen_samples = np.concatenate(gen_samples, 0)
        return (gen_samples[0:num_sample],gen_z[0:num_sample])


class Resnet(TwoStageVaeModel):
    def __init__(self, x, num_scale, block_per_scale=1, depth_per_block=2, kernel_size=3, base_dim=16, fc_dim=512, latent_dim=64, second_depth=3, second_dim=1024, l2_reg=.001):
        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size 
        self.base_dim = base_dim 
        self.fc_dim = fc_dim
        self.l2_reg = l2_reg
        super(Resnet, self).__init__(x, latent_dim, second_depth, second_dim)

    def build_encoder1(self):
        with tf.variable_scope('encoder'):
            dim = self.base_dim
            y = tf.layers.conv2d(self.x, dim, self.kernel_size, 1, 'same', name='conv0')
            for i in range(self.num_scale):
                y = scale_block(y, dim, self.is_training, 'scale'+str(i), self.block_per_scale, self.depth_per_block, self.kernel_size)

                if i != self.num_scale - 1:
                    dim *= 2
                    y = downsample(y, dim, self.kernel_size, self.l2_reg, 'downsample'+str(i))
            
            y = tf.reduce_mean(y, [1, 2])
            y = scale_fc_block(y, self.fc_dim, 'fc', 1, self.depth_per_block)
            
            #self.mu_z = tf.layers.dense(y, self.latent_dim*4)
            self.mu_z = tf.layers.dense(y, self.latent_dim)
            self.logsd_z = tf.layers.dense(y, self.latent_dim)
            self.sd_z = tf.exp(self.logsd_z)
            self.z = self.mu_z + tf.random_normal([self.batch_size, self.latent_dim]) * self.sd_z 

    def build_decoder1(self):
        desired_scale = self.x.get_shape().as_list()[1]
        scales, dims = [], []
        current_scale, current_dim = 2, self.base_dim 
        while current_scale <= desired_scale:
            scales.append(current_scale)
            dims.append(current_dim)
            current_scale *= 2
            current_dim = min(current_dim*2, 1024)
        assert(scales[-1] == desired_scale)
        dims = list(reversed(dims))

        with tf.variable_scope('decoder'):
            y = self.z 
            data_depth = self.x.get_shape().as_list()[-1]

            fc_dim = 2 * 2 * dims[0]
            y = tf.layers.dense(y, fc_dim, name='fc0')
            y = tf.reshape(y, [-1, 2, 2, dims[0]])

            for i in range(len(scales)-1):
                y = upsample(y, dims[i+1], self.kernel_size, self.l2_reg, 'up'+str(i))
                y = scale_block(y, dims[i+1], self.is_training, 'scale'+str(i), self.block_per_scale, self.depth_per_block, self.kernel_size)
            
            y = tf.layers.conv2d(y, data_depth, self.kernel_size, 1, 'same')
            self.x_hat = tf.nn.sigmoid(y)

### MAIN

import fid
    
def main():
    tf.reset_default_graph()
    # exp info
    exp_folder = os.path.join(args.output_path, args.dataset, args.exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model_path = os.path.join(exp_folder, 'checkp')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # dataset
    x, side_length, channels = load_dataset(args.dataset, args.root_folder)
    input_x = tf.placeholder(tf.uint8, [args.batch_size, side_length, side_length, channels], 'x')

    # model
    if args.network_structure != 'Resnet':
        model = eval(args.network_structure)(input_x, args.latent_dim, args.second_depth, args.second_dim, args.l2_reg)
    else:
        model = Resnet(input_x, args.num_scale, args.block_per_scale, args.depth_per_block, args.kernel_size, args.base_dim, args.fc_dim, args.latent_dim, args.second_depth, args.second_dim,args.l2_reg)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(exp_folder, sess.graph)
    saver = tf.train.Saver()

    # train model

    if not args.val:
        # first stage
        if True: #True to restore last checkpoint
            saver.restore(sess, os.path.join(model_path, 'stage2'))
            xin = x[:10000]
            _,_,img_recons = model.reconstruct(sess, xin) 
            seloss = np.mean(np.square(xin/ 255. - img_recons),axis = (1,2,3))
            mseloss = np.mean(seloss)
            gamma_x = np.sqrt(mseloss)
            print("mse: ", mseloss)
            mu_z, _ = model.extract_posterior(sess, xin)
            z_hat = model.reconstruct2(sess, mu_z)
            mseloss2 = np.mean(np.square(mu_z - z_hat), axis = (0,1))
            gamma_z = np.sqrt(mseloss2)
            print("mse2: ", mseloss2)
            
        else:
            mseloss = 1.
            gamma_x = 1.
            mseloss2 = 1.
            gamma_z = 1.
            
        num_sample = np.shape(x)[0]
        print('Num Sample = {}.'.format(num_sample))
        iteration_per_epoch = num_sample // args.batch_size
        
        for epoch in range(args.epochs):
            np.random.shuffle(x)
            lr = args.lr if args.lr_epochs <= 0 else args.lr * math.pow(args.lr_fac, math.floor(float(epoch) / float(args.lr_epochs)))
            epoch_loss = 0
            for j in range(iteration_per_epoch):
                image_batch = x[j*args.batch_size:(j+1)*args.batch_size]
                loss, bmseloss = model.step(1, image_batch, gamma_x,lr, sess, writer, args.write_iteration)
                epoch_loss += loss
                #print("mse: ", bmseloss)
                #we estimate mse as a weighted combination of the
                #the previous estimation and the minibatch mse
                mseloss = min(mseloss,mseloss*.99+bmseloss*.01)
                gamma_x = np.sqrt(mseloss)
            epoch_loss /= iteration_per_epoch
            print('Date: {date}\t'
                  'Epoch: [Stage 1][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch, args.epochs, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
            print("Gamma_x: ", gamma_x)
            print("mse: ", mseloss)
            
        saver.save(sess, os.path.join(model_path, 'stage1'))

        # second stage
        mu_z, sd_z = model.extract_posterior(sess, x)
        idx = np.arange(num_sample)
        for epoch in range(args.epochs2):
            np.random.shuffle(idx)
            mu_z = mu_z[idx]
            sd_z = sd_z[idx]
            lr = args.lr2 if args.lr_epochs2 <= 0 else args.lr2 * math.pow(args.lr_fac2, math.floor(float(epoch) / float(args.lr_epochs2)))
            epoch_loss = 0
            for j in range(iteration_per_epoch):
                mu_z_batch = mu_z[j*args.batch_size:(j+1)*args.batch_size]
                sd_z_batch = sd_z[j*args.batch_size:(j+1)*args.batch_size]
                z_batch = mu_z_batch + sd_z_batch * np.random.normal(0, 1, [args.batch_size, args.latent_dim])
                loss, bmseloss2 = model.step(2, z_batch, gamma_z, lr, sess, writer, args.write_iteration)
                epoch_loss += loss
                mseloss2 = min(mseloss2,mseloss2*.99+bmseloss2*.01)
                gamma_z = np.sqrt(mseloss2)
            epoch_loss /= iteration_per_epoch

            print('Date: {date}\t'
                  'Epoch: [Stage 2][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch, args.epochs2, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
        saver.save(sess, os.path.join(model_path, 'stage2'))
    else:
        saver.restore(sess, os.path.join(model_path, 'stage2'))
    
    x = x[0:10000]
    
    tf.reset_default_graph()
    zmean, zlogvar, img_recons = model.reconstruct(sess, x)

    img_gens1,_ = model.generate(sess, 10000, 1)
    img_gens2,gen_z = model.generate(sess, 10000, 2)

    x = x.astype("float32") / 255

    # computing FID can be expensive 
    if True:
        print("Rec FID: ", fid.get_fid(x, img_recons.copy()))
        print("Gen FID (1): ", fid.get_fid(x, img_gens1.copy()))
        print("Gen FID (2): ", fid.get_fid(x, img_gens2.copy()))
    
    #img_recons_sample = stich_imgs_2(x, img_recons)
    #img_gens1_sample = stich_imgs(img_gens1)
    #img_gens2_sample = stich_imgs(img_gens2)
    #plt.imsave(os.path.join(exp_folder, 'gen1_sample.jpg'), img_gens1_sample)
    #plt.imsave(os.path.join(exp_folder, 'gen2_sample.jpg'), img_gens2_sample)
    #plt.imsave(os.path.join(exp_folder, 'recon_sample.jpg'), img_recons_sample)
    
    print("MSE: ", np.mean(np.square(x[:10000] - img_recons.copy()), axis = (0,1,2,3)))
    
    zmeanvar = np.var(zmean, axis = 0)
    zlogvarmean = np.mean(np.exp(zlogvar), axis = 0)
    zsum = zmeanvar + zlogvarmean

    print("variance law = ", np.mean(zsum))

    count = 0
    for i in range(args.latent_dim):
        #print(zlogvarmean[i])
        if (zlogvarmean[i] > 0.8):
            count += 1
    print("Inactive var: ", count)

    def stich_imgs(x, img_per_row=10, img_per_col=10):
        x_shape = np.shape(x)
        assert(len(x_shape) == 4)
        output = np.zeros([img_per_row*x_shape[1], img_per_col*x_shape[2], x_shape[3]])
        idx = 0
        for r in range(img_per_row):
            start_row = r * x_shape[1]
            end_row = start_row + x_shape[1]
            for c in range(img_per_col):
                start_col = c * x_shape[2]
                end_col = start_col + x_shape[2]
                output[start_row:end_row, start_col:end_col] = x[idx]
                idx += 1
                if idx == x_shape[0]:
                    break
            if idx == x_shape[0]:
                break
        if np.shape(output)[-1] == 1:
            output = np.reshape(output, np.shape(output)[0:2])
        return output

    def stich_imgs_2(x_raw, x, img_per_row=10, img_per_col=2):
        x_shape = np.shape(x)
        assert(len(x_shape) == 4)
        output = np.zeros([img_per_col*x_shape[2], img_per_row*x_shape[1], x_shape[3]])
        idx = 0
        for r in range(img_per_row):
            start_row = r * x_shape[1]
            end_row = start_row + x_shape[1]
            output[ 0:x_shape[2], start_row:end_row] = x_raw[idx]
            idx += 1
            if idx == x_shape[0]:
                break
        idx = 0
        for r in range(img_per_row):
            start_row = r * x_shape[1]
            end_row = start_row + x_shape[1]
            output[x_shape[2]:2 * x_shape[2], start_row:end_row] = x[idx]
            idx += 1
            if idx == x_shape[0]:
                break
        if np.shape(output)[-1] == 1:
            output = np.reshape(output, np.shape(output)[0:2])
        return output

    for i in range(0,5):
        imgs = stich_imgs(img_gens2[i*100:(i+1)*100])
        plt.figure(figsize=(20,20))
        plt.axis("off")
        plt.imshow(imgs)
        plt.savefig("imgs"+str(i)+".png", bbox_inches='tight')
        plt.show()

##############################################

#dictionary => object (simulate argparse)


#    suggested configurations:
#                 cifar    celeba         
#    second_dim   2048     4096
#    num_scale    3        4
#    epochs       700      70
#    lr           .0001    .00005
#    lr_epochs    200      60
#    epochs2      1400     140
#    lr2          .0001    .00005
#    lr_epochs2   400      120
#    l2_reg       .0005    .001


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = {
    "root_folder"       : '../Tensor/',
    "output_path"       : './experiments',
    "exp_name"          : 'Exp_1',
    "dataset"           : 'celeba', #'cifar10',
    "gpu"               : 1,
    "network_structure" : 'Resnet',
    "batch_size"        : 100, 
    "write_iteration"   : 600,
    "latent_dim"        : 64,
    "second_dim"        : 4096, #use 2048 for cifar
    "second_depth"      : 3,
    "num_scale"         : 4, #was 3
    "block_per_scale"   : 1,
    "depth_per_block"   : 2,
    "kernel_size"       : 3,
    "base_dim"          : 32, 
    "fc_dim"            : 512,
    "epochs"            : 0, #use 700 for cifar
    "lr"                : 0.00005, #use .0001 for cifar
    "lr_epochs"         : 60, #use 200 for cifar
    "lr_fac"            : 0.5, 
    "epochs2"           : 0, #use 1400 for cifar
    "lr2"               : 0.00005, #use .0001 for cifar
    "lr_epochs2"        : 120, #use 1200 for cifar
    "lr_fac2"           : 0.5,
    "l2_reg"            : 0.001, #use 0.0005 for cifar
    "val"               : True
}
args = Struct(**args)

main()
