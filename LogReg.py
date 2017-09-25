import numpy as np
import tensorflow as tf
import random

nb_inp = 4
OR_X = []
OR_y = []
def make_samples(k):
    OR_X = []
    OR_y = []
    for i in range(k):
        a = random.randint(0,11)
        b = random.randint(0,a)
        c = random.randint(0,a)
        OR_X.append([b,c])
        OR_y.append([a])
    OR_X = np.array(OR_X)
    OR_y = np.array(OR_y)
    return (OR_X, OR_y)
    # print(XOR_X)
    # print(XOR_y)

(OR_X, OR_y) = make_samples(10)

# print(XOR_X)
# print(XOR_y)

X = tf.placeholder(dtype = tf.float32, shape = [None, 2])
W = tf.get_variable(name = 'W', dtype = tf.float32, initializer=tf.ones([2, 1]))
b = tf.get_variable(name = 'b', dtype = tf.float32, initializer= tf.zeros([1]))
unnormalized_prob = tf.add(tf.matmul(X, W), b)
y = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
y_ = tf.placeholder(shape = [i for i in y.shape], dtype = tf.float32)
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = unnormalized_prob, labels = y_)
train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.run(train_step, feed_dict={X : OR_X, y_ : OR_y})

(OR_X, OR_y) = make_samples(10000)

y_prob = sess.run(y, feed_dict = {X : OR_X})
print (OR_y,y_prob)


correct_pred = 0

for i in range(100):
    if (OR_y[i,0] > 5 and y_prob[i,0] >= 0.5) or (OR_y[i,0] < 5 and y_prob[i,0] < 0.5):
        correct_pred += 1

print(correct_pred/100)
