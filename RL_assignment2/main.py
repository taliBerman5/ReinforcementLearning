import gym
import numpy as np
import matplotlib.pyplot as plt
import time

gamma = 0.95
epsilon = 1
steps_amount = 1000000
steps_per_episode = 250
num_of_simulates = 100

action_map = {0: "move_left", 1: "move_down", 2: "move_right", 3: "move_up"}

grid_map = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]


def create_policy_small(Q, env):
    arrow = ['<', 'v', '>', '^']
    for i in range(4):
        print('')
        for j in range(4):
            location = j + i * 4
            a = np.argmax(Q[location])
            print(arrow[a], end='')
    print("")
    env.render()



def epsilon_greedy(env, Q_table, state, eps):
    if np.random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state, :])


def Q_learning(env, eps, alpha):
    i = 0
    policy_value = []
    policy_steps_value = []
    Q_table = np.zeros((env.observation_space.n, env.action_space.n))

    while i <= steps_amount:
        state = env.reset()
        for j in range(steps_per_episode):
            action = epsilon_greedy(env, Q_table, state, eps)
            s_tag, r, done, prob = env.step(action)
            Q_table[state, action] += alpha *(r + (gamma * np.max(Q_table[s_tag, :])) - Q_table[state, action])

            i += 1

            if i == 1000 or i == 3000 or i == 5000 or i % 10000 == 0:
                policy_steps_value.append(i)
                policy_value.append(eval_policy(env, Q_table))

            if done:
                if r == 1:  # decay epsilon only if reached to the goal
                     eps = max(0.01, eps * 0.99)
                break
            state = s_tag

    return Q_table, policy_value, policy_steps_value


def Q_learning_EGT(env, eps, alpha, lamda):
    i = 0
    policy_value = []
    policy_steps_value = []
    Q_table = np.zeros((env.observation_space.n, env.action_space.n))

    while i <= steps_amount:
        state = env.reset()
        EGT = np.zeros((env.observation_space.n, env.action_space.n))
        reached = {(0, 0)}
        reached.remove((0, 0))
        for j in range(steps_per_episode):
            action = epsilon_greedy(env, Q_table, state, eps)
            reached.add((state, action))
            EGT[state][action] += 1
            s_tag, r, done, prob = env.step(action)
            delta = r + (gamma * np.max(Q_table[s_tag, :])) - Q_table[state, action]

            for s_, a_ in reached:
                Q_table[s_, a_] += alpha * delta * EGT[s_][a_]
                EGT[s_][a_] = gamma * lamda * EGT[s_][a_]

            i += 1

            if i == 1000 or i == 3000 or i == 5000 or i % 10000 == 0:
                policy_steps_value.append(i)
                policy_value.append(eval_policy(env, Q_table))


            if done:
                if r == 1:  # decay epsilon only if reached to the goal
                     eps = max(0.01, eps * 0.99)
                break
            state = s_tag

    return Q_table, policy_value, policy_steps_value


def eval_policy(env, Q_table):
    total_rewards = 0
    sum_success = 0
    for i in range(num_of_simulates):
        state = env.reset()
        path_len = 0
        r = 0
        discount_factor = []
        rewards = []
        curr_reward = 0
        for j in range(steps_per_episode):
            action = np.argmax(Q_table[state, :])
            state, r, done, prob = env.step(action)

            discount_factor.append(gamma ** path_len)
            rewards.append(r)
            path_len += 1
            if done:
                break
        for ri in range(path_len):
            curr_reward += discount_factor[ri] * rewards[ri]
        total_rewards += curr_reward

        sum_success += r

    return total_rewards / num_of_simulates


def simulate(env, Q_table):
    sum_of_reward = 0
    steps = 0
    state = env.reset()
    done = False
    counter = 1
    goal_pos = "7,7"
    while not done:
        row = int(state / 8)
        column = state % 8
        agent_pos = str(row) + "," + str(column)
        action = np.argmax(Q_table[state, :])
        state, r, done, prob = env.step(action)
        sum_of_reward += r
        r = int(r)
        if r > 0:
            r = "+" + str(r)
        print(str(counter) + ".", agent_pos, goal_pos, action_map[action], grid_map[row][column], r)
        steps += 1
        counter += 1

    print("total steps:", steps)
    if sum_of_reward > 0:
        sum_of_reward = "+" + str(sum_of_reward)
    print("total rewards:", sum_of_reward, "\n")


def simulateRender(env, Q_table):
    state = env.reset()
    env.render()
    done = False
    while not done:
        state, r, done, prob = env.env.step(np.argmax(Q_table[state, :]))
        env.render()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # initialization
    env = gym.make('FrozenLake8x8-v1')
    #env = gym.make('FrozenLake-v1')
    env.render()
    alpha_arr = [0.03, 0.05, 0.03, 0.1]
    lamda_arr = [0.2, 0.1, 0.05, 0.1]
    Q_table_saver = []
    policy_value_saver = []
    policy_steps_value = []
    for i in range(4):
        startTime = time.perf_counter()
        Q_table_EGT, policy_value, policy_steps_value = Q_learning_EGT(env, epsilon, alpha_arr[i], lamda_arr[i])
        Q_table_saver.append(Q_table_EGT)
        policy_value_saver.append(policy_value)
        end_one_q_leraningTime = time.perf_counter()
        print("q_leraningTime took with: alpha = " +str(alpha_arr[i]) +" lambda ="+str(lamda_arr[i])  , (end_one_q_leraningTime - startTime) / 60, " minutes")

    plt.figure()
    for i in range(4):
        label_i = 'alpha = ' + str(alpha_arr[i]) + ', lambda = ' + str(lamda_arr[i])
        plt.plot(policy_steps_value, policy_value_saver[i], label=label_i)
    plt.legend()
    plt.title("Average Policy value with eligibility traces")
    plt.xlabel("steps")
    plt.ylabel("policy value")
    plt.show()

    Q_table, policy_value, policy_steps_value = Q_learning(env, epsilon, alpha_arr[0])

    plt.figure()
    plt.plot(policy_steps_value, policy_value_saver[1], label='eligibility traces')
    plt.plot(policy_steps_value, policy_value, label='WO eligibility traces')
    plt.legend()
    plt.title("Average Policy value of alpha:"+str(alpha_arr[0])+ "lambda:" + str(lamda_arr[0]))
    plt.xlabel("steps")
    plt.ylabel("policy value")
    plt.show()



    # simulateRender(env, Q_table_EGT)
    # for i in range(3):
    #     print("-----------------------------------------------------")
    #     simulate(env, Q_table_EGT)
