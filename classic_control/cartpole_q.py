"""CartPole Learning Agent using Q Learning"""


import gym
import numpy as np
import math
import matplotlib.pyplot as plt


def create_buckets(n, low, high):
    """Return list of n bucket ranges between low and high."""
    low = np.float64(low)
    high = np.float64(high)
    offset = (high - low) / n
    buckets = []
    bound = low
    while round(bound, 8) <= high:
        buckets.append(bound)
        bound = bound + offset
    return buckets


env = gym.make('CartPole-v0')
# Discretize continuous theta values into ranges of buckets
theta_bucket_num = 6
theta_buckets = create_buckets(theta_bucket_num, env.observation_space.low[2], env.observation_space.high[2])

# Discretize continuous theta_dot values into ranges of buckets
theta_dot_bucket_num = 3
theta_dot_buckets = create_buckets(theta_dot_bucket_num, -math.radians(50), math.radians(50))

Q = np.zeros((theta_bucket_num, theta_dot_bucket_num, env.action_space.n))


def get_explore_rate(epsilon, episode):
    # With decay
    # return max(epsilon, min(1, 1.0 - math.log10((episode + 1.0) / 25)))
    # Without decay
    return epsilon


def get_learning_rate(lr, episode):
    # With decay
    # return max(lr, min(0.5, 1.0 - math.log10((episode + 1.0) / 25)))
    # Without decay
    return lr


def get_bucket_index(value, buckets):
    """Returns index of bucket value falls into."""
    for i, bucket_limit in enumerate(buckets):
        if value <= bucket_limit:
            # Bucket upper limit found
            # Bucket index is upper limit index - 1
            return i - 1
    # Bucket not found
    return -1


def set_qvalue(state, action, value):
    """Set Q Value associated to state action pair to value."""
    theta = state[2]
    theta_dot = state[3]
    theta_bucket_index = get_bucket_index(theta, theta_buckets)
    theta_dot_bucket_index = get_bucket_index(theta_dot, theta_dot_buckets)
    Q[theta_bucket_index][theta_dot_bucket_index][action] = value


def get_qvalue(state, action):
    """Retrive Q Value associated to state.

    State is continuous so find discretized bucket in QTable.
    """
    theta = state[2]
    theta_dot = state[3]
    theta_bucket_index = get_bucket_index(theta, theta_buckets)
    theta_dot_bucket_index = get_bucket_index(theta_dot, theta_dot_buckets)
    return Q[theta_bucket_index][theta_dot_bucket_index][action]


def get_max_qvalue(state):
    """Return maximum q-value for given state."""
    return max([get_qvalue(state, action) for action in range(env.action_space.n)])


def get_action(state, epsilon):
    """Determine action to take based on current state.

    Follows epsilon greedy strategy to explore/exploit.
    """
    if np.random.random_sample() < epsilon:
        # Explore
        return env.action_space.sample()
    else:
        # Exploit
        return get_optimal_action(state)


def get_optimal_action(state):
    """Determine action to take that optimizes value given current state.

    When there are ties for maximum q values, randomly choose action.
    """
    max_q_value = get_max_qvalue(state)
    max_actions = []
    for action in range(env.action_space.n):
        if get_qvalue(state, action) == max_q_value:
            max_actions.append(action)
    return np.random.choice(max_actions)


def update(state, action, reward, next_state, learning_rate, discount_factor):
    """Update Q value following update rule.

    Q(s, a) <- (1 - learning_rate) * Q(s, a) + learning_rate * target
    where target is Bellmans = R(s, a, s') + discount * max(Q(s', a'))
    """
    target = reward + discount_factor * get_max_qvalue(next_state)
    update_value = (1 - learning_rate) * get_qvalue(state, action) + learning_rate * target
    set_qvalue(state, action, update_value)


def run(num_episodes=1, epsilon=5e-1, learning_rate=2e-1, discount_factor=9e-1, verbose=False, history=[]):
    """Run set trials with selected hyper parameters."""
    window = []
    solved = False
    for i in range(num_episodes):
        episode_reward = 0.0
        # Decay hyper params
        e = get_explore_rate(epsilon, i)
        lr = get_learning_rate(learning_rate, i)
        # Episode is begin to terminal
        state = env.reset()
        for t in range(200):
            # Time steps
            # Find next action and take step
            action = get_action(state, e)
            next_state, reward, done, _ = env.step(action)
            # Observe next state and update q values
            update(state, action, reward, next_state, lr, discount_factor)
            # Update state
            state = next_state
            episode_reward += reward
            if done:
                break
        # Rolling mean to determine solved
        window.append(episode_reward)
        if len(window) == 100:
            avg_reward = np.array(window).mean()
            history.append(avg_reward)
            if avg_reward > 195.0:
                # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
                print('Solved in {} episodes with average reward {}.'.format(i + 1, avg_reward))
                solved = True
                # demo()
                # return avg_reward
            # Slide window
            window.pop(0)

    if verbose and not solved:
        print('Not Solved with last window average reward of {}.'.format(avg_reward))

    return avg_reward


def demo():
    """Demo solved cartpole with stored QTable."""
    state = env.reset()
    for t in range(200):
        # Time steps
        env.render()
        # Only Exploit
        action = get_action(state, 0)
        next_state, reward, done, _ = env.step(action)
        # Update state
        state = next_state
        if done:
            break


def run_hyper_param_optim():
    """Find best hyper parameters."""
    global Q
    epsilons = [1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]
    learning_rates = [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]
    discount_factors = [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 99e-2]

    max_avg_reward = 0.0
    max_e = epsilons[0]
    max_lr = learning_rates[0]
    max_d = discount_factors[0]

    # Hyper parameter optimization
    for e in epsilons:
        for lr in learning_rates:
            for d in discount_factors:
                avg_reward = run(num_episodes=300, epsilon=e, learning_rate=lr, discount_factor=d)
                if avg_reward > max_avg_reward:
                    max_avg_reward = avg_reward
                    max_e = e
                    max_lr = lr
                    max_d = d
                # Reset Q Table between trials
                Q = np.zeros((theta_bucket_num, theta_dot_bucket_num, env.action_space.n))

    print('Max average reward: {}'.format(max_avg_reward))
    print('Max epsilon: {}'.format(max_e))
    print('Max learning rate: {}'.format(max_lr))
    print('Max discount factor: {}'.format(max_d))


def graph(history):
    """Graph rewards."""
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.grid()
    axs.plot(range(1, len(history) + 1), history)
    axs.set(xlabel='Episode', ylabel='Average Reward', title='Episode Rewards')
    plt.show()


# run_hyper_param_optim()
history = []
run(num_episodes=800, epsilon=1e-5, learning_rate=1e-1, discount_factor=99e-2, verbose=True, history=history)
graph(history)
