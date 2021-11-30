import os
import pickle
import utils

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io


class TrainOps(object):
    # 设置训练参数
    def __init__(self, model, batch_size=64, pretrain_epochs=50, train_feature_generator_iters=100000,
                 train_DIFA_iters=201,
                 mnist_dir='./data/mnist', svhn_dir='./data/svhn', log_dir='./logs', model_save_path='./model',
                 pretrained_feature_extractor='feature_extractor', pretrained_feature_generator='feature_generator',
                 DIFA_feature_extractor='DIFA_feature_extractor'):

        self.model = model
        self.batch_size = batch_size

        # Number of iterations for Step 0, 1, 2
        # Step 0,1,2中的训练 epoch
        self.pretrain_epochs = pretrain_epochs
        self.train_feature_generator_iters = train_feature_generator_iters
        self.train_DIFA_iters = train_DIFA_iters

        # Dataset directory
        # 数据集文件夹
        self.mnist_dir = mnist_dir
        self.svhn_dir = svhn_dir

        # 存储模型的位置
        self.model_save_path = model_save_path
        self.pretrained_feature_extractor = os.path.join(self.model_save_path, pretrained_feature_extractor)
        self.pretrained_feature_generator = os.path.join(self.model_save_path, pretrained_feature_generator)
        self.DIFA_feature_extractor = os.path.join(self.model_save_path, DIFA_feature_extractor)

        # 存储 log
        self.log_dir = log_dir

        # 正对Session进行设置 动态申请显存
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True


    # 加载 mnist 数据，返回图片和lable
    def load_mnist(self, image_dir, split='train'):
        print('Loading MNIST dataset.')

        image_file = 'train.pkl' if split == 'train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']

        return images, np.squeeze(labels).astype(int)

    # 加载 svhn  数据，返回图片和lable
    def load_svhn(self, image_dir, split='train'):
        print('Loading SVHN dataset.')

        image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels == 10)] = 0
        return images, labels

    # 训练 feature extractor
    def train_feature_extractor(self):

        print('Pretraining feature extractor.')

        images, labels = self.load_svhn(self.svhn_dir, split='train')
        test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph of model
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            # 定义一个写入summary的目标文件
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            t = 0

            for i in range(self.pretrain_epochs):

                print('Epoch', str(i))

                #以 64为batch进行训练
                for start, end in zip(list(range(0, len(images), self.batch_size)),
                                      list(range(self.batch_size, len(images), self.batch_size))):
                    t += 1
                    # 此次投喂的batch
                    feed_dict = {model.images: images[start:end], model.labels: labels[start:end]}
                    # 训练 ，feed_dict给使用placeholder创建出来的tensor赋值
                    sess.run(model.train_op, feed_dict)

                    #每100 个batch 步记录对应的准确率并打印
                    if t % 100 == 0:
                        #随机选取 1000个进行训练效果
                        rand_idxs = np.random.permutation(images.shape[0])[:1000]

                        summary, loss, accuracy = sess.run(fetches=[model.summary_op, model.loss, model.accuracy],
                                                           feed_dict={model.images: images[rand_idxs],
                                                                   model.labels: labels[rand_idxs]})
                        # 写入 summary
                        summary_writer.add_summary(summary, t)
                        #打印训练结果
                        print('Step: [%d/%d] loss: [%.4f] accuracy: [%.4f]' % (
                            t + 1, self.pretrain_epochs * len(images) / self.batch_size, loss, accuracy))

            print('Saving')
            # 保存训练好的模型
            saver.save(sess, self.pretrained_feature_extractor)

    def train_feature_generator(self):

        print('Training sampler.')
        # 加载 svhn 数据
        images, labels = self.load_svhn(self.svhn_dir, split='train')
        # 将lable转为one hot 形式
        labels = utils.one_hot(labels, 10)

        # build a graph
        model = self.model
        model.build_model()

        noise_dim = 100
        # epochs = 5000

        with tf.Session(config=self.config) as sess:

            # initialize variables
            tf.global_variables_initializer().run()

            # restore feature extractor trained on Step 0
            # 论文中的Es
            print('Loading pretrained feature extractor.')
            variables_to_restore = slim.get_model_variables(scope='feature_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_feature_extractor)
            print('Loaded')

            # 定义一个写入summary的目标文件
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            for step in range(self.train_feature_generator_iters):

                i = step % int(images.shape[0] / self.batch_size)

                images_batch = images[i * self.batch_size:(i + 1) * self.batch_size]
                labels_batch = labels[i * self.batch_size:(i + 1) * self.batch_size]
                # 输入近generator 中的noise
                noise = utils.sample_Z(self.batch_size, noise_dim, 'uniform')

                feed_dict = {model.noise: noise, model.images: images_batch, model.labels: labels_batch}

                # 训练 Step 1 中 的 Discriminator 和 Generator
                sess.run(model.d_train_op, feed_dict)
                sess.run(model.g_train_op, feed_dict)

                # 每100次打印一次训练情况
                if (step + 1) % 100 == 0:
                    #
                    avg_D_fake = sess.run(model.logits_fake, feed_dict)
                    avg_D_real = sess.run(model.logits_real, feed_dict)
                    summary, dl, gl = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print(('Step: [%d/%d] d_loss: %.6f g_loss: %.6f avg_d_fake: %.2f avg_d_real: %.2f ' \
                           % (
                               step + 1, self.train_feature_generator_iters, dl, gl, avg_D_fake.mean(),
                               avg_D_real.mean())))

            print('Saving.')
            saver.save(sess, self.pretrained_feature_generator)

    def train_DIFA(self):

        print('Adapt with DIFA')
        # build a graph
        model = self.model
        model.build_model()
        # 加载所需的数据，分为source data 和 target data
        source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
        target_images, _ = self.load_mnist(self.mnist_dir, split='train')
        target_test_images, target_test_labels = self.load_mnist(self.mnist_dir, split='test')

        with tf.Session(config=self.config) as sess:

            # Initialize weights
            tf.global_variables_initializer().run()

            # 加载 Step0 中训练的encoder ，即Es
            print('Loading pretrained encoder.')
            variables_to_restore = slim.get_model_variables(scope='feature_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_feature_extractor)
            # 加载 Step1 中训练的 Generator
            print('Loading pretrained S.')
            variables_to_restore = slim.get_model_variables(scope='feature_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_feature_generator)

            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print('Start training.')

            for step in range(self.train_DIFA_iters):

                i = step % int(source_images.shape[0] / self.batch_size)
                j = step % int(target_images.shape[0] / self.batch_size)

                # 为每个batch准备数据
                source_images_batch = source_images[i * self.batch_size:(i + 1) * self.batch_size]
                target_images_batch = target_images[j * self.batch_size:(j + 1) * self.batch_size]
                labels_batch = utils.one_hot(source_labels[i * self.batch_size:(i + 1) * self.batch_size], 10)
                noise = utils.sample_Z(self.batch_size, 100, 'uniform')

                feed_dict = {model.src_images: source_images_batch, model.trg_images: target_images_batch,
                             model.noise: noise, model.labels: labels_batch}

                # 训练 Ei
                sess.run(model.e_train_op, feed_dict)
                # 训练 D2
                sess.run(model.d_train_op, feed_dict)

                # 每50步打印和保存训练结果
                if (step + 1) % 50 == 0:

                    logits_real, logits_fake = sess.run([model.logits_real, model.logits_fake], feed_dict)

                    summary, e_loss, d_loss = sess.run([model.summary_op, model.e_loss, model.d_loss], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print(('Step: [%d/%d] e_loss: [%.6f] d_loss: [%.6f] e_real: [%.2f] e_fake: [%.2f]' \
                           % (step + 1, self.train_DIFA_iters, e_loss, d_loss, logits_real.mean(), logits_fake.mean())))

                    print('Evaluating.')
                    target_test_acc = 0.

                    # 在 target data上的表现
                    for target_test_labels_batch, target_test_images_batch in zip(
                            np.array_split(target_test_labels, 100), np.array_split(target_test_images, 100)):
                        feed_dict[self.model.trg_images] = target_test_images_batch
                        feed_dict[self.model.trg_labels_gt] = target_test_labels_batch
                        target_test_acc_tmp = sess.run(model.trg_accuracy, feed_dict)
                        target_test_acc += target_test_acc_tmp / 100.

                    print('target test accuracy: [%.3f]' % (target_test_acc))
            #储存模型
            print('Saving.')
            saver.save(sess, self.DIFA_feature_extractor)


if __name__ == '__main__':
    from model import Model

    model = Model(mode='train_feature_extractor', learning_rate=0.0003)
    op = TrainOps(model)
    op.train_feature_generator()
