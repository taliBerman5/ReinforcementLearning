learning_rate = 0.0002
nm = [64,64,32,16]  # number of neuron per lyer
import gym
import numpy as np
# import tensorflow as tf
import collections
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()

env = gym.make('Pendulum-v0')
np.random.seed(1)


class PolicyNetwork:
    def _init_(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            # lyer1
            self.W1 = tf.get_variable("W1", [self.state_size, nm[0]],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [nm[0]], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [nm[0], nm[1]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [nm[1]], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output1 = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # lyer2
            self.W11 = tf.get_variable("W11", [nm[1], nm[1]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b11 = tf.get_variable("b11", [nm[1]], initializer=tf.zeros_initializer())
            self.W22 = tf.get_variable("W22", [nm[1], nm[2]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b22 = tf.get_variable("b22", [nm[2]], initializer=tf.zeros_initializer())

            self.Z11 = tf.add(tf.matmul(self.output1, self.W11), self.b11)
            self.A11 = tf.nn.relu(self.Z11)
            self.output2 = tf.add(tf.matmul(self.A11, self.W22), self.b22)

            # lyer3
            self.W111 = tf.get_variable("W111", [nm[2], nm[2]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b111 = tf.get_variable("b111", [nm[2]], initializer=tf.zeros_initializer())
            self.W222 = tf.get_variable("W222", [nm[2], nm[3]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b222 = tf.get_variable("b222", [nm[3]], initializer=tf.zeros_initializer())

            self.Z111 = tf.add(tf.matmul(self.output2, self.W111), self.b111)
            self.A111 = tf.nn.relu(self.Z111)
            self.output3 = tf.add(tf.matmul(self.A111, self.W222), self.b222)

            # lyer4
            self.W14 = tf.get_variable("W14", [nm[3], nm[3]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b14 = tf.get_variable("b14", [nm[3]], initializer=tf.zeros_initializer())
            self.W24 = tf.get_variable("W24", [nm[3], self.action_size],
                                       initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b24 = tf.get_variable("b24", [self.action_size], initializer=tf.zeros_initializer())

            self.Z14 = tf.add(tf.matmul(self.output3, self.W14), self.b14)
            self.A14 = tf.nn.tanh(self.Z14)
            self.output = tf.add(tf.matmul(self.A14, self.W24), self.b24)

            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class Val_net:
    def _init_(self, state_size, learning_rate, name='val_net'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.out_tder = tf.placeholder(tf.float32, name="out_tder")

            # layer 1
            self.W1 = tf.get_variable("W1", [self.state_size, nm[0]],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [nm[0]], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [nm[0], nm[1]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [nm[1]], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output1 = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # layer 2
            self.W12 = tf.get_variable("W12", [nm[1], nm[1]], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b12 = tf.get_variable("b12", [nm[1]], initializer=tf.zeros_initializer())
            self.W22 = tf.get_variable("W22", [nm[1], 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b22 = tf.get_variable("b22", [1], initializer=tf.zeros_initializer())

            self.Z12 = tf.add(tf.matmul(self.output1, self.W12), self.b12)
            self.A12 = tf.nn.relu(self.Z12)
            self.output = tf.add(tf.matmul(self.A12, self.W22), self.b22)

            self.s_m_d = tf.reduce_mean(tf.squared_difference(self.output, self.delta))

            self.loss = tf.reduce_mean(tf.abs(self.out_tder) * self.s_m_d)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyperparameters
actions = [-2.0 + i * 0.2 for i in range(21)]
state_size = 3
action_size = len(actions)  # env.action_space.n
max_episodes = 50000
max_steps = 501
discount_factor = 0.95

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)
Val_agent = Val_net(state_size, learning_rate)
# set Val agent network
# agent = DQNAgent()


# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = -1000.0
    last_average_rewards =-1000
    #saver.restore(sess, r"/content/models1/2917__-149.88616120817989/model.ckpt")

    for episode in range(max_episodes):
        gama_i = 1
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []
        action = 0

        for step in range(max_steps):
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step([actions[action]])
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(
                Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            # Compute Rt for each time-step t and update the network's weights
            #        for t, transition in enumerate(episode_transitions):
            # total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt
            v_tag = sess.run(Val_agent.output, {Val_agent.state: state})
            v_tag_stag = 0 if done else sess.run(Val_agent.output, {Val_agent.state: next_state})

            delta_targ = reward + discount_factor * v_tag_stag
            delta_eror = (delta_targ - v_tag) * gama_i

            feed_dict = {Val_agent.state: state, Val_agent.delta: delta_targ, Val_agent.out_tder: delta_eror}
            # Val_agent.train(state,delta)
            _, loss = sess.run([Val_agent.optimizer, Val_agent.loss], feed_dict)

            feed_dict = {policy.state: state, policy.R_t: delta_eror, policy.action: action_one_hot}
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

            # policy.train(action_one_hot,delta)

            gama_i = discount_factor * gama_i
            state = next_state
            ########################################
            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > -170 and average_rewards > last_average_rewards +3:
                    last_average_rewards =  average_rewards
                    path_model = os.path.join("/content/models1/",str(episode)+ "__"+str(last_average_rewards)+"/model.ckpt" )
                    save_path = saver.save(sess, path_model)
                    if last_average_rewards > -140:
                      solved = True
                      print(' Solved at episode: ' + str(episode))
                break

            if solved:
                break
        if solved:
            break
reward_perepisod_baceline = episode_rewards