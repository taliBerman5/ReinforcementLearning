import gym
import numpy as np
import time
import matplotlib.pyplot as plt


gamma = 1
alpha = 0.01
steps_amount = 500000
steps_per_episode = 200
num_of_simulates = 100

center_p = [-0.9, -0.5, -0.1, 0.2]
center_v = [-0.05, -0.02, -0.01, 0, 0.01, 0.03, 0.05, 0.07]


combination_c = [[p, v] for p in center_p for v in center_v]
action_map = {0: "la", 1: "na", 2: "ra"}


def epsilon_greedy(env, state, w, eps):
    if np.random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q_estimate(state, w))


def q_estimate(state, W):
    return W @ teta_calc(state)


def teta_calc(state):
    x = np.array(state) - combination_c
    diag_inv = np.diag([1 / 0.04, 1 / 0.0004])
    tetas =  np.exp(-0.5 * x @ diag_inv @ x.T)
    return np.diag(tetas)

def q_estimate_action(state, W, action):
    return q_estimate(state, W)[action]

def q_grad(state):
    return teta_calc(state)



def Q_learning(env, eps):
    start = time.perf_counter()
    i = 0
    policy_value = []
    policy_steps_value = []
    W = np.random.rand(3, len(combination_c))
    while i <= steps_amount:
        state = env.reset()
        for j in range(steps_per_episode):
            action = epsilon_greedy(env, state, W, eps)
            s_tag, r, done, _ = env.step(action)
            td_error = r + gamma * np.max(q_estimate(s_tag, W)) - q_estimate_action(state, W, action)
            W[action] += alpha * td_error * q_grad(state)

            i += 1

            if i == 1000 or i == 3000 or i == 5000 or i % 10000 == 0:
                policy_steps_value.append(i)
                policy_value.append(eval_policy(env, W))

            if done:
                eps = max(0.01, eps * 0.99)
                break
            state = s_tag
    endFirst = time.perf_counter()
    print("took ", (endFirst - start) / 60, " minutes")
    return W, policy_value, policy_steps_value


def eval_policy(env, W):
    total_rewards = 0
    for i in range(num_of_simulates):
        state = env.reset()
        path_len = 0
        discount_factor = []
        rewards = []
        curr_reward = 0
        for j in range(steps_per_episode):
            action = np.argmax(q_estimate(state, W))
            state, r, done, _ = env.step(action)

            discount_factor.append(gamma ** path_len)
            rewards.append(r)
            path_len += 1
            if done:
                break
        for ri in range(path_len):
            curr_reward += discount_factor[ri] * rewards[ri]
        total_rewards += curr_reward

    return total_rewards / num_of_simulates


def simulate(env, W):
    sum_of_reward = 0
    steps = 0
    state = env.reset()
    done = False
    counter = 1
    goal = "0.5, 0"
    while not done:
        agent = str(state[0]) + "," + str(state[1])
        action = np.argmax(q_estimate(state, W))
        state, r, done, _ = env.step(action)
        sum_of_reward += r
        print(str(counter) + ".", agent, goal, action_map[action], str(r))
        steps += 1
        counter += 1

    print("total steps:", steps)
    print("total rewards:", str(sum_of_reward), "\n")

def simulateRender(env, W):
    state = env.reset()
    env.render()
    done = False
    while not done:
        state, r, done, _ = env.step(np.argmax(q_estimate(state, W)))
        env.render()




def plots(env):
    W, policy_value, policy_steps_value = Q_learning(env, eps=1)

    plt.figure()
    plt.plot(policy_steps_value, policy_value, label='policy value')
    plt.legend()
    plt.title("Average Policy value")
    plt.xlabel("steps")
    plt.ylabel("policy value")
    plt.show()

    simulateRender(env, W)
    for i in range(3):
        print("-----------------------------------------------------")
        simulate(env, W)



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.reset()
    plots(env)



