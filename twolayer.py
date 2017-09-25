import numpy as np
import tensorflow as tf
import random

#Creating Test data
W = [np.random.randint(low = 10,size=(1,10))]  #[10,10]
OW = [np.random.randint(low=10,size=(10,1))]
def test_data(batch_size):
    X_Data = []
    Y_Data = []
    for x in range(batch_size):
        Array = np.random.randint(low=10,size=(10,1)) #[10,1]
        X_Data.append(Array)
        Array2 = np.dot(Array,W)
        Array3 = np.dot(Array2,OW)
        Y_Data.append(Array3) #[10,1]
    X_Data = np.asarray(X_Data)
    Y_Data = np.asarray(Y_Data)
    X_Data = np.reshape(X_Data,(batch_size*10,1))
    Y_Data = np.reshape(Y_Data,(batch_size*10,1))
    return X_Data, Y_Data

###Building layers
rate = .5
#Input Layer
X = tf.placeholder(name = 'x', dtype=tf.float32, shape = [None, 1])
W1 = tf.get_variable(name = 'W1', dtype=tf.float32, initializer=tf.ones([1,10]))
b = tf.get_variable(name = 'b', dtype = tf.float32, initializer= tf.zeros([1]))
#Hidden Layer
HL = tf.add(tf.matmul(X,W1), b)
W2 = tf.get_variable(name = 'W2', dtype=tf.float32, initializer=tf.ones([10,1]))
b2 = tf.get_variable(name = 'b2', dtype = tf.float32, initializer= tf.zeros([1]))
#Output Layer
output = tf.add(tf.matmul(HL,W2),b2)
y = tf.placeholder(name = 'y', dtype=tf.float32, shape = [i for i in output.shape])
#Calculating gradient descent
cost = tf.reduce_mean(tf.square(y - output))
train_step = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(cost)
#Setup
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#Training Model
X_Data, Y_Data = test_data(1000)
sess.run(train_step, feed_dict={X : X_Data, y : Y_Data})
#Testing Model
X_Data, Y_Data = test_data(100)
y_prob = sess.run(y, feed_dict = {X : X_Data,y: Y_Data})
correct_pred = 0
for i in range(100):
    if (np.absolute(Y_Data[i,0] - y_prob[i,0]) < .001):
        print(Y_Data[i,0])
        print(y_prob[i,0])
        correct_pred += 1

print(correct_pred/100)
