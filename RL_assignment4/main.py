import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy

gamma = 0.9
alpha = 0.01
beta = 0.01
sigma = 0.4
sigma1 = sigma2 = 0.3
sigma3 = 0.4

steps_amount = 500000
steps_per_episode = 200
num_of_simulates = 100

center_cos_sin = [[0, 1], [0.05, 0.95], [-0.05, 0.95], [0.1, 0.9], [-0.1, 0.9], [0.15, 0.92], [-0.15, 0.92], [0.4, 0.8], [-0.4, 0.8], [0.8, 0.4], [-0.8, 0.4], [0, -1], [0.8, 0.4], [-0.8, 0.4], [0.7, -0.7], [-0.7, -0.7]]
center_speed = [-6.0, -2.0, -0.5, -0.1, 0, 0.1, 0.5, 2.0, 6.0]


centers = np.array([[c[0], c[1], speed] for c in center_cos_sin for speed in center_speed])
print(len(centers))


def sample(state, theta):
    a = np.random.normal(mu_calc(state, theta), sigma)
    # print("a " ,a)
    return [a]

def mu_calc(state, theta):
    mu = features_calc(state).T @ theta
    # print(mu)
    return mu

def grad_log_pi(state, action, theta):
    mu = mu_calc(state, theta)
    features = features_calc(state)
    return ((action[0]-mu) * features)/sigma**2



def v_estimate(state, W):
    return W @ features_calc(state)


def features_calc(state):
    x = state - centers
    diag_inv = np.diag([1 / sigma1, 1 / sigma2, 1 / sigma3])
    tetas = np.exp(-0.5 * x @ diag_inv @ x.T)
    return np.diag(tetas)


def v_grad(state):
    return features_calc(state)



def actor_critic(env):
    i = 0
    total_return = 0
    policy_value = []
    policy_steps_value = []
    W = np.random.rand(len(centers))
    theta = np.random.rand(len(centers))
    while i <= steps_amount:
        state = env.reset()
        I = 1
        for j in range(steps_per_episode):
            action = sample(state, theta)
            s_tag, r, done, _ = env.step(action)
            # print(state)

            delta = r + gamma * v_estimate(s_tag, W) - v_estimate(state, W)
            theta += I * alpha*grad_log_pi(state, action, theta)*delta
            W += I * beta * delta * v_grad(state)
            total_return += r
            i += 1

            # if i == 1000 or i == 3000 or i == 5000 or i % 10000 == 0:
            #     print(i)
            #     policy_steps_value.append(i)
            #     policy_value.append(eval_policy(env, theta))


            if done:
                # simulateRender(env, theta)
                print(total_return)
                total_return = 0
                break
            state = s_tag
            I = gamma*I
    return theta, policy_value, policy_steps_value


def eval_policy(env, theta):
    simulateRender(env, theta)
    total_rewards = 0
    for i in range(num_of_simulates):
        state = env.reset()
        path_len = 0
        discount_factor = []
        rewards = []
        curr_reward = 0
        for j in range(steps_per_episode):
            action = sample(state, theta)
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


def simulate(env, theta):
    sum_of_reward = 0
    steps = 0
    state = env.reset()
    done = False
    counter = 1
    goal = "0.5, 0"
    while not done:
        agent = str(state[0]) + "," + str(state[1])
        action = sample(state, theta)
        state, r, done, _ = env.step(action)
        sum_of_reward += r
        print(str(counter) + ".", agent, goal, str(r))
        steps += 1
        counter += 1

    print("total steps:", steps)
    print("total rewards:", str(sum_of_reward), "\n")

def simulateRender(env, theta):
    state = env.reset()
    env.render()
    done = False
    while not done:
        state, r, done, _ = env.step(sample(state, theta))
        env.render()




def plots(env):
    theta, policy_value, policy_steps_value = actor_critic(env)

    plt.figure()
    plt.plot(policy_steps_value, policy_value, label='policy value')
    plt.legend()
    plt.title("Average Policy value")
    plt.xlabel("steps")
    plt.ylabel("policy value")
    plt.show()

    simulateRender(env, theta)
    # for i in range(3):
    #     print("-----------------------------------------------------")
    #     simulate(env, theta)



if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    env.reset()
    # print(env.action_space)
    # for i in [(-2+j*0.01) for j in range(190)]:
    #     a = env.action_space.sample()
    #     print(i)
    #     s = env.step([i])
    #     print(s)
    #     env.render()
    plots(env)



