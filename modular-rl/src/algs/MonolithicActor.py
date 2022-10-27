import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MonolithicPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_action,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super(MonolithicPolicy, self).__init__()
        self.num_agents = 1
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.max_action = max_action
        self.batch_size = batch_size
        self.max_children = max_children
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.monolithic_max_agent = args.monolithic_max_agent

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim * self.monolithic_max_agent, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim * self.monolithic_max_agent),
            nn.Tanh(),
        ).to(device)

    def forward(self, state, synergy, mode="train", env_name=None):
        self.clear_buffer()
        batch_size = state.shape[0]
        self.input_state = state.reshape(batch_size, -1)
        inpt = F.pad(self.input_state,
                     pad=[0, self.state_dim * (self.monolithic_max_agent - self.num_agents)],
                     value=0)
        self.action = self.actor(inpt)[:,:self.num_agents*self.action_dim]
        self.action = self.max_action * self.action

        return torch.squeeze(self.action)


    def change_morphology(self, parents):
        self.parents = parents
        self.num_agents = sum([len(x) for x in parents])
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents

    def clear_buffer(self):
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

