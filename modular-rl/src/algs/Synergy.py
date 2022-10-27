import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import AffinityPropagation
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_AGG = 4

class GroupAttention(nn.Module):
    def __init__(self, d_input, d_model, d_output, dropout=0.8, n_embeds = 10, args=None):
        super(GroupAttention, self).__init__()
        self.d_model = d_model
        self.args = args
        self.d_input = d_input
        embed_dim = 64
        self.embeds = nn.ModuleList([
            nn.Embedding(n_embeds, embed_dim) for _ in range(d_input)
        ])

        self.aggregate = nn.Linear(embed_dim * d_input, d_model * D_AGG)

        self.embeds2 = nn.ModuleList([
            nn.Embedding(n_embeds, embed_dim) for _ in range(d_input)
        ])

        self.aggregate2 = nn.Linear(embed_dim * d_input, d_model * D_AGG)

    def get_embed(self, context):
        key = torch.cat([
            self.embeds[i](context[:, :, i]) for i in range(self.d_input)
        ], dim=-1)
        key = self.aggregate(key)  # [1,n,d]
        return key

    def forward(self, env_name, context, weights=False):
        key = torch.cat([
            self.embeds[i](context[:, :, i]) for i in range(self.d_input)
        ], dim=-1)
        key = self.aggregate(key)  # [1,n,d]
        assert self.args.synergy_action_dim == 1
        SD = self.args.synergy_action_dim * D_AGG
        N = key.shape[1]
        E = key.shape[2] // SD
        key = key.reshape(N, E, SD)
        k1 = key[:, 0].permute(1, 0)  #
        k2 = key[:, 1:]
        scores = (k2.permute(2, 0, 1) @ k2.permute(2, 1, 0)) / E  # [1/SD,N,E]@[1/SD,E,N]=[1/SD,N,N]
        scores = scores - torch.diag_embed(k1) # [1/SD,N,N]

        return scores

class Synergy(nn.Module):
    def __init__(self, args=None):
        super(Synergy, self).__init__()
        self.args = args
        embed_num = 3
        distance, embedding = utils.getStructureInfo(args.graphs, embed_num=embed_num)

        self.robot_joint_embeds = dict()
        for k, v in embedding.items():
            emb = embedding[k]
            self.robot_joint_embeds[k] = torch.tensor(emb).to(device).long().unsqueeze(0)

        self.affinity = {
            n: np.exp(-distance[n]) for n in distance.keys()
        }
        self.preference = {
            name: None for name in self.affinity.keys()
        }

        self.synergy_weights = dict()
        self.synergy_cluster = dict()
        self.synergy_cluster_center = dict()
        self.update_synergy_weights()

        d_model = args.d_model
        self.n_synergy = args.max_num_agents

        self.action_attn = GroupAttention(embed_num, d_model, self.n_synergy,
                                          n_embeds = args.max_num_agents, args=args)


    def update_synergy_weights(self, env_name = None, update_preference = None):
        if env_name is not None:
            self.preference[env_name] = update_preference
            updated_names = [env_name]
        else:
            updated_names = self.affinity.keys()

        for name in updated_names:
            affinity = self.affinity[name]
            preference = self.preference[name]
            ap = AffinityPropagation(preference=preference, affinity='precomputed',
                                     random_state=0, verbose=False).fit(affinity)

            labels = ap.labels_
            centers = ap.cluster_centers_indices_
            if (labels == labels[0]).sum() > 1:
                labels[0] = len(centers)
                centers = np.append(centers, 0)

            weights = F.one_hot(torch.tensor(labels)).float().to(device)
            weights[:,[labels[0]]] = 1.0
            self.synergy_weights[name] = weights
            self.synergy_cluster[name] = labels
            self.synergy_cluster_center[name] = centers

    def get_mask(self, env_name):
        w = self.get_synergy(env_name, delay=True).detach() # [n,K]
        w = F.normalize(w, dim=1, p=2)
        cos = torch.mm(w, w.t())
        mask = cos.log()

        return mask

    def get_merge_weight(self, env_name):
        w = self.get_synergy(env_name, delay=True).detach().t()  # [K,n]
        w = F.normalize(w, dim=1, p=1)
        return w

    def get_action_weight(self, env_name, delay=False):
        inpt = self.robot_joint_embeds[env_name]  # [1,n,K]
         # [n,n]
        w = self.get_synergy(env_name, delay=True).detach()  # [n,K]
        c = self.get_syngery_center(env_name, delay=True) # [K]
        N,K = w.shape
        a = self.action_attn(env_name, inpt, weights=True) #/K #/N # [1/SD,N,N]
        output = a[:,c].permute(0,2,1) # [1/SD,K,N] => [1/SD,N,K]
        output = output * w
        return output # [1/SD,N,K]

    def get_embed(self, env_name):
        inpt = self.robot_joint_embeds[env_name]  # [1,n,K]
        e = self.action_attn.get_embed(inpt) # [1,n,d]
        return e

    def get_synergy(self, env_name, delay=False):
        w = self.synergy_weights[env_name] # [n,K]
        return w

    def get_syngery_center(self, env_name, delay=False):
        c = self.synergy_cluster_center[env_name] # [K]
        return c

    def get_info(self):
        info = dict()
        info.update({
            f'synery_cluster_{n}': self.synergy_cluster[n].reshape(1,-1) for n in self.args.envs_train_names
        })
        info.update({
            f'synery_cluster_center_{n}': self.synergy_cluster_center[n].reshape(1, -1) for n in self.args.envs_train_names
        })
        return info


    def change_morphology(self, graph):
        pass
