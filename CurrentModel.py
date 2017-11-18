import tensorflow as tf
import numpy as np
import scipy.misc
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
import timeit
import math
import os
import json
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.utils import shuffle

#Built by Nhat Hoang Pham, Herbert Turner

# Make this a multiple of 64
n_sample = 1300

class FirstModel:
    def __init__(self, inp_w, inp_h, inp_d,training, keep_prob = 0.5,):
        with tf.device('/device:CPU:0'):
            self._is_training = tf.placeholder(tf.bool)
            self._X = tf.placeholder(shape = [None, inp_w, inp_h, inp_d], dtype = tf.float32)
            # self._y = tf.placeholder(tf.int64, shape = [None])

            # First Convolutional Layer:
            self._inp_norm = tf.layers.batch_normalization(self._X, axis = 1, training = self._is_training)
            self._Wconv1 = tf.get_variable("Wconv1", initializer = tf.ones([3,3, inp_d, 32]))
            self._bconv1 = tf.get_variable("bconv1", initializer = tf.zeros(32))
            self._z1 = tf.nn.conv2d(self._inp_norm, self._Wconv1, strides = [1, 1, 1, 1], padding = 'VALID') + self._bconv1
            self._h1 = tf.layers.batch_normalization(tf.nn.relu(self._z1), axis = 1, training = self._is_training)

            # Second convolutional layer:
            self._Wconv2 = tf.get_variable("Wconv2", initializer = tf.ones([3,3, 32, 64]))
            self._bconv2 = tf.get_variable("bconv2", initializer = tf.zeros(64))
            self._z2 = tf.nn.conv2d(self._h1, self._Wconv2, strides = [1, 1, 1 ,1], padding = 'VALID') + self._bconv2
            self._h2 = tf.layers.batch_normalization(tf.nn.relu(self._z2), axis = 1, training = self._is_training)
            self._h2_max_pool = self.max_pool_2x2(self._h2)

            #Third convolutional layer:
            self._Wconv3 = tf.get_variable("Wconv3", initializer = tf.ones([3,3,64,64]))
            self._bconv3 = tf.get_variable("bconv3", initializer = tf.zeros(64))
            self._z3 = tf.nn.conv2d(self._h2_max_pool, self._Wconv3, strides = [1,1,1,1], padding = 'VALID') + self._bconv3
            self._h3 = tf.layers.batch_normalization(tf.nn.relu(self._z3), axis = 1, training = self._is_training)

            #Fourth convolutional layer"
            self._Wconv4 = tf.get_variable("Wconv4", initializer = tf.ones([3,3,64,64]))
            self._bconv4 = tf.get_variable("bconv4", initializer = tf.zeros(64))
            self._z4 = tf.nn.conv2d(self._h3, self._Wconv4, strides = [1,1,1,1], padding='VALID') + self._bconv4
            self._h4 = tf.layers.batch_normalization(tf.nn.relu(self._z4), axis = 1, training = self._is_training)
            self._h4_max_pool = self.max_pool_2x2(self._h4)

            #Fifth convolutional layer"
            self._Wconv5 = tf.get_variable("Wconv5", initializer = tf.ones([3,3,64,64]))
            self._bconv5 = tf.get_variable("bconv5", initializer = tf.zeros(64))
            self._z5 = tf.nn.conv2d(self._h4_max_pool, self._Wconv5, strides = [1,1,1,1], padding='VALID') + self._bconv5
            self._h5 = tf.layers.batch_normalization(tf.nn.relu(self._z5), axis = 1, training = self._is_training)

            #Sixth convolutional layer"
            self._Wconv6 = tf.get_variable("Wconv6", initializer = tf.ones([3,3,64,64]))
            self._bconv6 = tf.get_variable("bconv6", initializer = tf.zeros(64))
            self._z6 = tf.nn.conv2d(self._h5, self._Wconv6, strides = [1,1,1,1], padding='VALID') + self._bconv6
            self._h6 = tf.layers.batch_normalization(tf.nn.relu(self._z6), axis = 1, training = self._is_training)
            self._h6_max_pool = self.max_pool_2x2(self._h6)

            #Seventh convolutional layer"
            self._Wconv7 = tf.get_variable("Wconv7", initializer = tf.ones([3,3,64,64]))
            self._bconv7 = tf.get_variable("bconv7", initializer = tf.zeros(64))
            self._z7 = tf.nn.conv2d(self._h6_max_pool, self._Wconv7, strides = [1,1,1,1], padding='VALID') + self._bconv7
            self._h7 = tf.layers.batch_normalization(tf.nn.relu(self._z7), axis = 1, training = self._is_training)

            #Eighth convolutional layer"
            self._Wconv8 = tf.get_variable("Wconv8", initializer = tf.ones([3,3,64,128]))
            self._bconv8 = tf.get_variable("bconv8", initializer = tf.zeros(128))
            self._z8 = tf.nn.conv2d(self._h7, self._Wconv8, strides = [1,1,1,1], padding='VALID') + self._bconv8
            self._h8 = tf.layers.batch_normalization(tf.nn.relu(self._z8), axis = 1, training = self._is_training)
            self._h8_max_pool = self.max_pool_2x2(self._h8)
            self._h8_max_pool_flat = tf.reshape(self._h8_max_pool, [-1,3200], 'h8_max_pool_flat')



            # First fully-connected layer:
            self._W1 = tf.get_variable("W1", initializer = tf.ones([3200,1600]))
            self._b1= tf.get_variable("b1", initializer= tf.zeros(1600))
            self._fc1 = tf.matmul(self._h8_max_pool_flat, self._W1) + self._b1
            self._a2 = tf.nn.relu(self._fc1)
            self._fc1_normalized = tf.layers.batch_normalization(self._a2, axis = 1, training = self._is_training)

            # Dropout:
            #self._a2_dropout = tf.nn.dropout(self._a2, keep_prob)


            # Second fully-connected layer:
            self._W2 = tf.get_variable("W2", initializer = tf.ones([1600,1]))
            self._b2 = tf.get_variable("b2", initializer = tf.zeros([1]))
            self._a3 = tf.matmul(self._fc1_normalized, self._W2) + self._b2
            self._op = tf.layers.batch_normalization(self._a3, axis = 1, training = self._is_training)
            self.op = tf.sigmoid(self._op)

    def ret_op(self):
        return self._op

    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.round(self.op), self._y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training

        # counter
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                feed_dict = {self._X: Xd[idx, :],
                             self._y: yd[idx],
                             self._is_training: training_now}
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                # aggregate performance stats
                losses.append(loss * actual_batch_size)
                correct += np.sum(corr)

                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                          .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct, e + 1))
            # if plot_losses:
            #     plt.plot(losses)
            #     plt.grid(True)
            #     plt.title('Epoch {} Loss'.format(e + 1))
            #     plt.xlabel('minibatch number')
            #     plt.ylabel('minibatch loss')
            #     plt.show()
        self.saver = tf.train.Saver()
        self.saver.save(self._sess, r"C:\Users\cwessner\Documents\IP\classifiers" + str(i))
        return total_loss, total_correct


    # Define a max pool layer with size 2x2, stride of 2 and same padding.
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # Predict:
    def predict(self, X):
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                tf.global_variables_initializer().run()
                ans = sess.run(self._op, feed_dict = {self._X : X, self._is_training : True})
                return ans

    # Train:
    def train(self, X, y):
        self._y = tf.placeholder(tf.float32, shape = [None,1])
        self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self._op, labels = self._y))
        # print(self._mean_loss, self.op, self._op)
        self._optimizer = tf.train.AdamOptimizer(1e-4)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        # self.config=tf.ConfigProto()
        # # config.gpu_options.per_process_gpu_memory_fraction=0.98
        # self.config.gpu_options.allocator_type="BFC"
        # self.config.log_device_placement=True
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        print('Training Mole Classifier for 10 epochs' )
        self.run_model(self._sess, self._op, self._mean_loss, X, y, 10, 16, 1, self._train_step)
        self.saver = tf.train.Saver()
        self.saver.save(self._sess, r"C:\Users\cwessner\Documents\IP\classifier_final")


    def evaluate (self, X, y):
        self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 16)
#Original Image Size: 4160 x 3120
def readx(directory):
    images = []
    for root, dirs, files in os.walk(directory, topdown=False):
        x=0
        for name in files:
            if x < n_sample:
                if(name.find(".json")==-1):
                    image_data = scipy.misc.imread(directory + name,mode='RGB')
                    images.append(image_data)
                    x+=1
    return np.array(images)
def resized(images):
    resized_images = np.ones([len(images),128,128,3], dtype= np.float32)
    for x in range(len(images)):
        resized_images[x] = resize(images[x],(128,128,3))

    return resized_images
def ready(directory,length):
    benign_malignant = np.zeros([length])
    for root, dirs, files in os.walk(directory, topdown=False):
        x=0
        for name in files:
            if x < n_sample:
                if(name.find(".json")>-1):
                    with open(directory + name) as json_data:
                        d = json.load(json_data)
                        if(d['meta']['clinical']['benign_malignant']=="malignant"):
                            benign_malignant[x] = 1
                        x+=1
    return benign_malignant

#Training Network#
    #X_train1 = downsample(read(r"C:\Users\user\Documents\HackNEHS2017\Hacknehs2017\Datasets\\"))

# X, y = make_classification(n_classes=2, class_sep=2, weights=[0.3, 0.7],
#                            n_informative=3, n_redundant=1, flip_y=0,
#                            n_features=20, n_clusters_per_class=1,
#                            n_samples=80, random_state=10)
# print(X.shape,y.shape)

X = resized(readx(r"C:\Users\cwessner\Documents\IP\ISIC_allfinal\\"))
X_reshaped = X.reshape(len(X),len(X[0])*len(X[0][0])*len(X[0][0][0]))
y = ready(r"C:\Users\cwessner\Documents\IP\ISIC_allfinal\\",len(X))
indices = np.random.choice(n_sample,(int)(n_sample*.8))
X_train = []
y_train = []
X_val = []
y_val = []
for i in indices:
    X_train.append(X[i])
    y_train.append(y[i])
for i in range(n_sample):
    if i not in indices:
        X_val.append(X[i])
        y_val.append(y[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=0)
print(y_train)
# X_train_reshaped = X_train.reshape(len(X_train),len(X_train[0])*len(X_train[0][0])*len(X_train[0][0][0]))
# Oversampling the minority class using the SMOTE algorithm:
# sm = SMOTE(kind = 'regular', k_neighbors=1)
# X_train_resampled, y_train_resampled = sm.fit_sample(X_train_reshaped, y_train)
# X_train_resampled = X_train_resampled.reshape(len(X_train_resampled),400,400,3)
y_train = y_train.reshape(len(y_train),1)
covmodel = FirstModel(128,128,3,True)
print('Training')
covmodel.train(X_train,y_train)
print('Validation')
covmodel.evaluate(X_val,y_val)
