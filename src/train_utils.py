import numpy as np
from matplotlib import pyplot as plt

from src.stock_env import BufferMultiple


def fit_agent(agent, env_stock_train, stock_list, w_size=50):
    total_reward = 0
    state = env_stock_train.reset()

    stock_window = BufferMultiple(
        stock_list=stock_list,
        init_values=env_stock_train.data[stock_list].iloc[:w_size].values,
        size=w_size
    )
    state_normalized = stock_window.normalize(stock_window.get_values())
    print('state shape:', np.array(state_normalized).shape)

    while True:
        action = list(agent.get_action(state_normalized))
        next_state, reward, terminal, done, _, _ = env_stock_train.step(action)
        total_reward += reward

        price = next_state[0]
        stock_window.push(price)
        next_state_normalized = stock_window.normalize(stock_window.get_values())
        agent.fit(state_normalized, action, reward, done, next_state_normalized)

        if terminal or done:
            print('break')
            break
        state_normalized = next_state_normalized

    return total_reward


def agent_trading(agent, env_stock_test, stock_list, w_size=50):
    total_reward = 0
    state = env_stock_test.reset()
    stock_window = BufferMultiple(
        stock_list=stock_list,
        init_values=env_stock_test.data[stock_list].iloc[:w_size].values,
        size=w_size
    )
    state_normalized = stock_window.normalize(stock_window.get_values())

    while True:
        action = agent.get_action(state_normalized, do_noise=False)
        next_state, reward, terminal, done, _, _ = env_stock_test.step(action)
        total_reward += reward

        price = next_state[0]
        stock_window.push(price)
        next_state_normalized = stock_window.normalize(stock_window.get_values())

        if terminal or done:
            print('break')
            break
        state_normalized = next_state_normalized

    return total_reward


def rewards_plot(train_rewards, val_rewards, save=False, path=None):
    assert len(train_rewards) == len(val_rewards), "unmatched len of train and val rewards story"
    episode = len(train_rewards) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['blue', 'orange']
    labels = ['train', 'validation']
    for ax, rewards, color, label in zip(axes, [train_rewards, val_rewards], colors, labels):
        ax.set_title('episode: {}, \nlast total reward: {:.3f}%'.format(
            episode, rewards[-1] * 100
        ))
        ax.plot(rewards, color=color, lw=3, label=label, marker='0', markerfacecolor='white')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('total reward')

    if save and path:
        plt.savefig(path)
    plt.show()
