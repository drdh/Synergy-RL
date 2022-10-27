import torch
import torch.nn as nn

from .Transformer import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

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
        super(TransformerPolicy, self).__init__()
        self.num_agents = 1
        self.action = [None] * self.num_agents
        self.input_state = [None] * self.num_agents
        self.max_action = max_action
        self.batch_size = batch_size
        self.max_children = max_children
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = TransformerModel(
            self.state_dim,
            action_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            args.dropout_rate,
            condition_decoder=args.condition_decoder_on_features,
            transformer_norm=args.transformer_norm,
            used_by='policy',
            args=args,
        ).to(device)

    def forward(self, state, synergy, mode="train", env_name=None):
        self.clear_buffer()
        batch_size = state.shape[0]

        self.input_state = state.reshape(batch_size, self.num_agents, -1).permute(
            1, 0, 2
        )
        self.action = self.actor(self.input_state, synergy, mode, env_name=env_name)
        self.action = self.max_action * self.action

        # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)
        self.action = self.action.permute(1, 0, 2)

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
