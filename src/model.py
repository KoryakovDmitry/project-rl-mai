import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

import random
from copy import deepcopy
from collections import deque
from .utils import num_scale


class CNN2d(nn.Module):
    def __init__(self, input_dim: int = 50, out_channels: List[int] = [16, 32, 64], kernel_size: int = 3,
                 is_tanh: bool = True, stack_info=False,
                 dim_list: List[int] = [128, 32], output_dim: int = 10):
        super().__init__()
        self.w_size = input_dim
        self.stack_info = stack_info
        # self.cnn_layers = nn.Sequential(
        #    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2),
        #    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2),
        # )

        """
        CNN 30 stocks arch: (Conv.1d(channels = 32, kernel_size=3))
        self.cnn_2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=9),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )
        """
        # self.cnn_2d = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2),
        #     nn.ReLU(),
        # )

        self.cnn_1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_channels[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.modules_cnn = nn.ModuleList()
        # prev = (input_dim - (8 + 6 + 4 + 4 + 1) - (2)) * 64  # 64 * 11 #out_channels * (input_dim - (kernel_size - 1))
        prev = out_channels[2]
        if stack_info:  # доп информация о числе акций и размере портфеля
            self.modules_cnn.append(nn.Linear(prev + output_dim * 2, prev))
            self.modules_cnn.append(nn.ReLU())
        for i, dim in enumerate(dim_list + [output_dim]):
            self.modules_cnn.append(nn.Linear(prev, dim))
            if i < len(dim_list):
                self.modules_cnn.append(nn.ReLU())
            prev = dim

        self.tanh = nn.Tanh() if is_tanh else None

    def forward(self, x, num_stocks=None, hmax=None, stock_prices=None, balance=None):
        # unsqueeze_dim = 1 if len(x.shape) > 2 else 0
        # x = self.cnn_layers(x.unsqueeze(unsqueeze_dim)).unsqueeze(unsqueeze_dim)
        # print('1shhape:', x.shape)
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
        # x = x.unsqueeze(unsqueeze_dim)
        # print('2shhape:', x.shape)
        # x = self.cnn_2d(x)
        # print('3shhape:', x.shape)
        # x = x.squeeze()
        # print('4shhape:', x.shape)
        x = self.cnn_1d(x)
        # x = x.unsqueeze(unsqueeze_dim)
        # print('5shhape:', x.shape)
        # x = self.flatten(x)
        # print('6shhape:', x.shape)
        if self.stack_info:
            batch_size, _ = x.shape
            x = torch.cat([
                x,
                torch.FloatTensor(num_scale(num_stocks, div=hmax)).expand(batch_size, len(num_stocks)),
                torch.FloatTensor(num_scale(balance / np.array(stock_prices), div=hmax)).expand(batch_size,
                                                                                                len(stock_prices))
            ], dim=1)

        x = x.transpose(1, 2);
        # print('7shhape:', x.shape)
        for module in self.modules_cnn:
            # print('cnn shhape:', x.shape)
            x = module(x)

        if self.tanh:
            x = self.tanh(x)
        # print('7shhape:', x.shape)
        x = x.mean(dim=1)
        return x



# Ornstein–Uhlenbeck process (Процесс Орнштейна – Уленбека)
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPG:
    def __init__(
            self,
            state_dim,
            action_dim,
            action_max,
            device,
            policy,
            policy_type,
            multiple_stocks,
            kwargs=None,
            gamma=0.99,
            tau=1e-3,
            batch_size=64,
            q_model_lr=1e-3,
            pi_model_lr=1e-4,
            noise_decrease=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.device = device
        self.policy = policy
        self.policy_type = policy_type
        self.kwargs = kwargs
        self.multiple_stocks = multiple_stocks
        if kwargs is None:
            self.pi_model = policy(state_dim, 400, 300, action_dim, is_tanh=True).to(
                device
            )
            self.q_model = policy(
                state_dim + action_dim, 400, 300, 1, is_tanh=False
            ).to(device)
        else:
            self.pi_model = policy(**kwargs, input_dim=state_dim, is_tanh=True).to(
                device
            )
            self.q_model_input_dim = (
                state_dim + action_dim if policy_type not in ["lstm"] else state_dim
            )
            self.q_model = policy(
                **kwargs, input_dim=self.q_model_input_dim, is_tanh=True
            ).to(device)
        self.pi_target_model = deepcopy(self.pi_model)

        self.q_target_model = deepcopy(self.q_model)
        self.noise = OUNoise(action_dimension=self.kwargs["output_dim"])
        self.noise_threshold = 1
        self.noise_decrease = noise_decrease
        self.noise_min = 0.01
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.pi_optimazer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_model_lr)
        self.q_optimazer = torch.optim.Adam(self.q_model.parameters(), lr=q_model_lr)
        self.pi_model_lr = pi_model_lr
        self.q_model_lr = q_model_lr

    def get_action(self, state, do_noise=True):
        state = torch.FloatTensor(state).to(self.device)
        _action = (
                self.pi_model(state).detach().data.cpu().numpy()
                + self.noise_threshold * self.noise.sample() * do_noise
        )
        return self.action_max * _action

    def update_target_model(self, target_model, model, optimazer, loss):
        optimazer.zero_grad()
        loss.backward()
        optimazer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data
            )

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(
                lambda x: x.to(self.device), map(torch.FloatTensor, zip(*batch))
            )
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            pred_next_actions = self.action_max * self.pi_target_model(next_states)
            # print(next_states.shape, pred_next_actions.shape)
            # if self.multiple_stocks:
                # pred_next_actions = pred_next_actions.unsqueeze(1)
            # print("ddd", next_states.shape, pred_next_actions.unsqueeze(dim=1).shape)
            next_states_and_pred_next_actions = torch.cat(
                (next_states, pred_next_actions.unsqueeze(dim=1)), dim=1
            )
            targets = rewards + self.gamma * (1 - dones) * self.q_target_model(
                next_states_and_pred_next_actions
            )
            if self.multiple_stocks:
                actions = actions.unsqueeze(1)
            states_and_actions = torch.cat((states, actions), dim=1)
            temp = self.q_model(states_and_actions) - targets.detach()
            q_loss = torch.mean(
                (targets.detach() - self.q_model(states_and_actions)) ** 2
            )
            self.update_target_model(
                self.q_target_model, self.q_model, self.q_optimazer, q_loss
            )

            pred_actions = self.action_max * self.pi_model(states)
            if self.multiple_stocks:
                pred_actions = pred_actions.unsqueeze(1)
            # print('sss', states.shape, pred_actions.shape)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            pi_loss = -torch.mean(self.q_model(states_and_pred_actions))
            self.update_target_model(
                self.pi_target_model, self.pi_model, self.pi_optimazer, pi_loss
            )

    def save(self, dir="DDPG_agent"):
        print(os.path.join(os.getcwd(), dir, f"pi_model.pkl"))
        # torch.save(self.pi_model.state_dict(), os.path.join(os.getcwd(), dir, f"pi_model.pkl"))
        # torch.save(self.q_model.state_dict(), os.path.join(os.getcwd(), dir, f"q_model.pkl"))

    def load(self, dir):
        if self.kwargs is None:
            self.pi_model = self.policy(
                self.state_dim, 400, 300, self.action_dim, is_tanh=True
            ).to(self.device)
            self.q_model = self.policy(
                self.state_dim + self.action_dim, 400, 300, 1, is_tanh=False
            ).to(self.device)
        else:
            self.pi_model = self.policy(
                **self.kwargs, input_dim=self.state_dim, is_tanh=True
            ).to(self.device)
            self.q_model = self.policy(
                **self.kwargs, input_dim=self.q_model_input_dim, is_tanh=True
            ).to(self.device)

        self.pi_model.load_state_dict(torch.load(f"{dir}/pi_model.pkl"))
        self.pi_target_model = deepcopy(self.pi_model)
        self.pi_optimazer = torch.optim.Adam(
            self.pi_model.parameters(), lr=self.pi_model_lr
        )

        self.q_model.load_state_dict(torch.load(f"{dir}/q_model.pkl"))
        self.q_target_model = deepcopy(self.q_model)
        self.q_optimazer = torch.optim.Adam(
            self.q_model.parameters(), lr=self.q_model_lr
        )
