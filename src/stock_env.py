from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


class BufferMultiple:
    def __init__(self, stock_list, init_values=None, size=30, eps=1e-6):
        self.size = size
        self.stock_list = stock_list
        self.values = [] if init_values is None else list(list(i) for i in init_values[:size])
        self.eps = eps

    def normalize(self, values):
        if len(values) > 0:
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            # print(f"std: {std}, mean: {mean}")
            return list(list(i) for i in ((np.array(values) - mean) / (std + self.eps)))

    def push(self, values_list):
        self.values.append(values_list)
        if len(self.values) > self.size:
            self.values.pop(0)

    def set_values(self, values):
        self.values = values

    def get_values(self):
        return self.values


class StockTradingMultipleEnv(gym.Env):
    def __init__(
            self,
            data: pd.DataFrame,
            stock_list: List[str],
            env_name: str = "train",
            hmax: int = 100,
            initial_amount: float = 1e5,
            window_size: int = 30,
            action_space: int = 1,
            reward_scaling: float = 1e-5,
    ):
        self.env_name = env_name
        self.data = data
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.balance = initial_amount
        self.portfolio_size = None
        self.portfolio_size_story = []
        self.stock_list = stock_list
        self.num_stocks = [0] * len(stock_list)
        self.stock_price = None
        self.state_space = 1 + 1 + 1
        # state space = 1(stock price) + 1(portfolio size) + 1(num_stocks in portfolio)
        self.action_space = action_space
        self.actions_buffer = []

        self.state_story = []
        self.reward_story = []
        self.reward_scaling = reward_scaling

        assert window_size <= len(data), "not enough data to fill stock window"
        self.window_size = window_size
        self.timestep = None
        self.total_timesteps = len(data["date"].unique())
        self.terminal = False
        self.done = False

        self._initiate_state()

    def _sell_stock(self, idx, action: int) -> None:
        assert action >= 0, "action can't be negative"
        sell_count = min(action, self.num_stocks[idx])
        self.num_stocks[idx] -= sell_count
        self.balance += sell_count * self.stock_prices[idx]

    def _buy_stock(self, idx, action: int) -> None:
        assert action >= 0, "action can't be negative"
        available_amount = self.balance // self.stock_prices[idx]
        buy_count = min(action, available_amount)
        self.num_stocks[idx] += buy_count
        self.balance -= buy_count * self.stock_prices[idx]

    def act(self, actions: List[int]):
        sell_idxes = np.where(actions <= 0)[0]
        for idx in sell_idxes:
            self._sell_stock(idx, np.abs(actions[idx]))

        buy_idxes = np.where(actions > 0)[0]
        for idx in buy_idxes:
            self._buy_stock(idx, actions[idx])

    def _make_plot(self):
        plt.figure(figsize=(10, 8))
        plt.title(f"Portfolio size during time:, env - {self.env_name}:")
        print(
            f"timestep: {self.timestep}, portfolio story len: {len(self.portfolio_size_story)}"
        )
        plt.plot(  # self.data['date'].iloc[self.window_size:self.timestep+1].values,
            self.portfolio_size_story, color="orange", lw=3
        )
        plt.xlabel("time")
        plt.ylabel("portfolio size")
        plt.grid(True)
        plt.show()

    def step(self, actions: List[float]):
        self.terminal = self.timestep > self.total_timesteps - 2
        self.done = self.portfolio_size <= 0

        if self.terminal or self.done:
            # self._make_plot()
            pass
        else:
            assert len(actions) == len(self.stock_list), "not enough actions"
            actions = np.array(
                [
                    int(np.clip(action * self.hmax, -self.hmax, self.hmax))
                    for action in actions
                ]
            )
            self.act(actions)
            # print(f"action: {action}")
            self.actions_buffer.append(actions)
            # if action > 0:
            #    self._buy_stock(action)
            # else:
            #    self._sell_stock(np.abs(action))

            begin_asset = self.portfolio_size
            self._update_state()
            end_asset = self.portfolio_size
            self.reward = (end_asset - begin_asset) * self.reward_scaling
            self.reward_story.append(self.reward)

        return self.state, self.reward, self.terminal, self.done, False, {}

    def reset(self):
        self.terminal = False
        self.done = False
        self.balance = self.initial_amount
        self.num_stocks = [0] * len(self.stock_list)
        self.portfolio_size_story.clear()
        self.actions_buffer.clear()
        self.state_story.clear()
        self.reward_story.clear()
        self._initiate_state()
        # print(f'num_stocks: {self.num_stocks}, balance: {self.balance}, portfolio: {self.portfolio_size}')
        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        # print('init state')
        self.reward = None
        self.timestep = self.window_size
        self.portfolio_size = self.initial_amount
        self.portfolio_size_story.append(self.portfolio_size)
        self.stock_prices = list(self.data[self.stock_list].iloc[self.timestep].values)
        self.state = [
            self.stock_prices,
            self.num_stocks,
            self.portfolio_size,
            self.balance,
        ]
        self.state_story.append(self.state)

    def _update_state(self):
        self.timestep += 1
        self.stock_prices = list(self.data[self.stock_list].iloc[self.timestep].values)
        self.portfolio_size = self.balance + np.sum(
            np.array(self.num_stocks) * self.stock_prices
        )
        self.portfolio_size_story.append(self.portfolio_size)
        self.state = [
            self.stock_prices,
            self.num_stocks,
            self.portfolio_size,
            self.balance,
        ]
        self.state_story.append(self.state)

    def get_date(self):
        return self.data.iloc[self.timestep]["date"]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
