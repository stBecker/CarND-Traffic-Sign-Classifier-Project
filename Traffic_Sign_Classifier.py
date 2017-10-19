
# coding: utf-8


# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "../data/train.p"
validation_file="../data/valid.p"
testing_file = "../data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import pandas as pd
import numpy as np

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train)) # reduce(np.union1d, (y_test, y_valid, y_train)).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


dfy=pd.DataFrame(y_train)
dfyv=pd.DataFrame(y_valid)
dfyt=pd.DataFrame(y_test)



train_data_balanced=[]
for class_id in range(n_classes):
    idx = dfy[dfy[0] == class_id].sample(2000, replace=True).index
    x = X_train[idx]
    y = y_train[idx]
    assert np.unique(y).size == 1
    train_data_balanced.append((y, x))
    
X_train_balanced = np.concatenate([x for y, x in train_data_balanced])
y_train_balanced = np.concatenate([y for y, x in train_data_balanced])


def normalize_pixels(x):
    return (x - 128.) / 128



def rgb2gray(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

X_train_norm = normalize_pixels(X_train)
X_valid_norm = normalize_pixels(X_valid)
X_test_norm = normalize_pixels(X_test)


X_train_norm_gray = rgb2gray(X_train_norm)
X_valid_norm_gray = rgb2gray(X_valid_norm)
X_test_norm_gray = rgb2gray(X_test_norm)


X_train_norm_gray = X_train_norm_gray.reshape(*X_train_norm_gray.shape, 1)
X_valid_norm_gray = X_valid_norm_gray.reshape(*X_valid_norm_gray.shape, 1)
X_test_norm_gray = X_test_norm_gray.reshape(*X_test_norm_gray.shape, 1)



import tensorflow as tf

EPOCHS = 12
BATCH_SIZE = 128



from tensorflow.contrib.layers import flatten


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
#         accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_hot: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
        #print(sess.run(one_hot_y, feed_dict={x: batch_x, y: batch_y}))
    return total_accuracy / num_examples




# final model

with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 43)

    keep_prob = tf.placeholder(tf.float32)

    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    rate = 0.001

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    prediction = tf.argmax(logits, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()



from sklearn.utils import shuffle



# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# # config.gpu_options.per_process_gpu_memory_fraction = 0.7
# config.log_device_placement = True


X=X_train_balanced.copy()
X=normalize_pixels(X)
Y=y_train_balanced.copy()


valid_acc = []
train_acc = []

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X, Y = shuffle(X, Y)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X[offset:end], Y[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(X_valid_norm, y_valid)
        training_accuracy = evaluate(X, Y)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()
        valid_acc.append(validation_accuracy)
        train_acc.append(training_accuracy)
        
    saver.save(sess, './traffic_sign_classifier3')
    print("Model saved")
