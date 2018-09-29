import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import batch_norm

class Network:
    def __init__(self, session, action_count, state_length, lr, batch_size, trace_length, hidden_size, scope):
        self.session = session
        self.train_batch_size = batch_size
        self.state_length = state_length
        self.trace_length_size = trace_length
        self.hidden_size = hidden_size
        
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        self.bn_params = {
            'is_training' : self.is_training,
            'decay' : 0.99,
            'updates_collections' : None
        }

        self.state = tf.placeholder(tf.float32, shape=[None, self.state_length])
        #self.state = tf.Print(self.state, [self.state], message='state:', summarize=100)

        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        self.flat = tf.contrib.layers.fully_connected(self.state, hidden_size, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, normalizer_params=self.bn_params)
        #self.flat = tf.Print(self.flat, [self.flat], message='flat:', summarize=100)

        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

        self.fc_reshape = tf.reshape(self.flat, [self.batch_size, self.train_length, hidden_size])
        #self.fc_reshape = tf.Print(self.fc_reshape, [self.fc_reshape], message='fc_reshape:', summarize=100)


        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.fc_reshape, cell=self.cell, dtype=tf.float32,
                                                     initial_state=self.state_in, scope=scope+'_rnn')

        self.rnn = tf.reshape(self.rnn, shape=[-1, hidden_size])
        #self.rnn = tf.Print(self.rnn, [self.rnn], message='rnn:', summarize=100)

        self.q = tf.contrib.layers.fully_connected(self.rnn, action_count, activation_fn=tf.nn.relu, normalizer_fn=batch_norm, normalizer_params=self.bn_params)
        #self.q = tf.Print(self.q, [self.q], message='q:', summarize=100)

        self.best_a = tf.argmax(self.q, 1)

        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_count, dtype=tf.float32)
        self.q_chosen = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot), axis=1)

        self.loss = tf.losses.mean_squared_error(self.q_chosen, self.target_q)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.95, epsilon=0.01)

        self.train_step = self.optimizer.minimize(self.loss)

    def learn(self, state, target_q, state_in, action):
        feed_dict = {self.state: state, self.target_q: target_q, self.train_length: self.trace_length_size,
                     self.batch_size: self.train_batch_size, self.state_in: state_in, self.actions: action, self.is_training: True}
        l, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
        return l

    def get_q(self, state, state_in):
        return self.session.run(self.q, feed_dict={self.state: state, self.train_length: self.trace_length_size,
                                                   self.batch_size: self.train_batch_size, self.state_in: state_in, self.is_training: True})

    def get_1q(self, state, state_in):
        #print('state:',state)
        res = self.session.run(self.q, feed_dict={self.state: state, self.train_length: 1,
                                                   self.batch_size: 1, self.state_in: state_in, self.is_training: False})
        return res

    def get_best_action(self, state, state_in):
        return self.session.run([self.best_a, self.rnn_state], feed_dict={self.state: [state], self.train_length: 1,
                                                                          self.batch_size: 1, self.state_in: state_in, self.is_training: False})

    def get_cell_state(self, state, state_in):
        return self.session.run(self.rnn_state, feed_dict={self.state: [state], self.train_length: 1,
                                                           self.state_in: state_in, self.batch_size: 1, self.is_training: False})
