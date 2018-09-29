import tensorflow as tf
import tensorflow.contrib.slim as slim


class Network:
    def __init__(self, session, action_count, state_length, lr, batch_size, trace_length, hidden_size, scope):
        self.session = session
        self.train_batch_size = batch_size
        self.state_length = state_length
        self.trace_length_size = trace_length
        self.hidden_size = hidden_size

        self.state = tf.placeholder(tf.float32, shape=[None, self.state_length])
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        self.flat = tf.contrib.layers.legacy_fully_connected(x=self.state, num_output_units=hidden_size)

        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

        self.fc_reshape = tf.reshape(self.flat, [self.batch_size, self.train_length, hidden_size])
        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.fc_reshape, cell=self.cell, dtype=tf.float32,
                                                     initial_state=self.state_in, scope=scope+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, hidden_size])

        self.q = slim.fully_connected(self.rnn, action_count, activation_fn=None)

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
                     self.batch_size: self.train_batch_size, self.state_in: state_in, self.actions: action}
        l, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
        return l

    def get_q(self, state, state_in):
        return self.session.run(self.q, feed_dict={self.state: state, self.train_length: self.trace_length_size,
                                                   self.batch_size: self.train_batch_size, self.state_in: state_in})

    def get_1q(self, state, state_in):
        res = self.session.run(self.q, feed_dict={self.state: state, self.train_length: 1,
                                                   self.batch_size: 1, self.state_in: state_in})
        self.session.run(tf.Print(input_=self.state, data=[self.state], message='state:', summarize=10000))
        self.session.run(tf.Print(input_=self.flat, data=[self.flat], message='flat:', summarize=10000))
        self.session.run(tf.Print(input_=self.fc_reshape, data=[self.fc_reshape], message='fc_reshape:', summarize=10000))
        self.session.run(tf.Print(input_=self.rnn, data=[self.rnn], message='rnn:', summarize=10000))
        self.session.run(tf.Print(input_=self.q, data=[self.q], message='q:', summarize=10000))
        return res

    def get_best_action(self, state, state_in):
        return self.session.run([self.best_a, self.rnn_state], feed_dict={self.state: [state], self.train_length: 1,
                                                                          self.batch_size: 1, self.state_in: state_in})

    def get_cell_state(self, state, state_in):
        return self.session.run(self.rnn_state, feed_dict={self.state: [state], self.train_length: 1,
                                                           self.state_in: state_in, self.batch_size: 1})
