import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriticMonolithicPolicy(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            msg_dim,
            batch_size,
            max_children,
            disable_fold,
            td,
            bu,
            args=None,
    ):
        super(CriticMonolithicPolicy, self).__init__()
        self.num_agents = 1
        self.x1 = [None] * self.num_agents
        self.x2 = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.input_action = [None] * self.num_agents
        self.msg_down = [None] * self.num_agents
        self.msg_up = [None] * self.num_agents
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.monolithic_max_agent = args.monolithic_max_agent

        self.critic1 = nn.Sequential(
            nn.Linear( (self.state_dim + action_dim) * self.monolithic_max_agent, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim * self.monolithic_max_agent)
        ).to(device)

        self.critic2 = nn.Sequential(
            nn.Linear((self.state_dim + action_dim) * self.monolithic_max_agent, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim * self.monolithic_max_agent)
        ).to(device)

    def forward(self, state, action, synergy, env_name):
        self.clear_buffer()
        batch_size = state.shape[0]

        self.input_state = state.reshape(batch_size, self.num_agents, -1)
        self.input_action = action.reshape(batch_size, self.num_agents, -1)

        inpt = torch.cat([self.input_state, self.input_action], dim=2).reshape(batch_size, -1)
        inpt = F.pad(inpt,
                     pad=[0, (self.state_dim + self.action_dim) * (self.monolithic_max_agent - self.num_agents)],
                     value = 0
                     )
        self.x1 = self.critic1(inpt)[:, :self.num_agents].squeeze(-1)
        self.x2 = self.critic2(inpt)[:, :self.num_agents].squeeze(-1)
        return self.x1, self.x2

    def Q1(self, state, action, synergy, env_name):
        self.clear_buffer()
        batch_size = state.shape[0]
        self.input_state = state.reshape(batch_size, self.num_agents, -1)
        self.input_action = action.reshape(batch_size, self.num_agents, -1)
        inpt = torch.cat([self.input_state, self.input_action], dim=2).reshape(batch_size, -1)
        inpt = F.pad(inpt,
                     pad=[0, (self.state_dim + self.action_dim) * (self.monolithic_max_agent - self.num_agents)],
                     value=0
                     )
        self.x1 = self.critic1(inpt)[:, :self.num_agents].squeeze(-1)
        return self.x1


    def clear_buffer(self):
        self.x1 = [None] * self.num_agents
        self.x2 = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.input_action = [None] * self.num_agents
        self.msg_down = [None] * self.num_agents
        self.msg_up = [None] * self.num_agents
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        self.parents = parents
        self.num_agents = sum([len(x) for x in parents])
        self.msg_down = [None] * self.num_agents
        self.msg_up = [None] * self.num_agents
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents