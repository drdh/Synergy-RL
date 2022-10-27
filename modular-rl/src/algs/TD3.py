# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
from __future__ import print_function

import random
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F

from .ModularActor import ActorGraphPolicy
from .ModularCritic import CriticGraphPolicy
from .TransformerActor import TransformerPolicy
from .TransformerCritic import CriticTransformerPolicy
from .MonolithicActor import MonolithicPolicy
from .MonolithicCritic import CriticMonolithicPolicy
from .Synergy import Synergy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):
    def __init__(self, args):

        self.args = args
        random.seed(self.args.seed)
        if args.actor_type == "transformer":
            actor = TransformerPolicy
        elif args.actor_type == "smp":
            actor = ActorGraphPolicy
        elif args.actor_type == "monolithic":
            actor = MonolithicPolicy
        else:
            raise NotImplementedError

        self.actor = actor(
            args.agent_obs_size,
            1,
            args.msg_dim,
            args.batch_size,
            args.max_action,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(device)
        self.actor_target = actor(
            args.agent_obs_size,
            1,
            args.msg_dim,
            args.batch_size,
            args.max_action,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(device)
        if args.critic_type == "transformer":
            critic = CriticTransformerPolicy
        elif args.critic_type == "smp":
            critic = CriticGraphPolicy
        elif args.critic_type == "monolithic":
            critic = CriticMonolithicPolicy
        else:
            raise NotImplementedError

        self.critic = critic(
            args.agent_obs_size,
            1,
            args.msg_dim,
            args.batch_size,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(device)
        self.critic_target = critic(
            args.agent_obs_size,
            1,
            args.msg_dim,
            args.batch_size,
            args.max_children,
            args.disable_fold,
            args.td,
            args.bu,
            args,
        ).to(device)
        self.synergy = Synergy(args).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.synergy_optimizer = torch.optim.Adam(self.synergy.parameters(), lr=args.lr)

        self.ctr = {
            n : defaultdict(int) for n in self.args.envs_train_names
        }

        self.models2eval()

    def change_morphology(self, graph):
        self.actor.change_morphology(graph)
        self.actor_target.change_morphology(graph)
        self.critic.change_morphology(graph)
        self.critic_target.change_morphology(graph)
        self.synergy.change_morphology(graph)

    def select_action(self, state, env_name):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state, self.synergy, "inference", env_name=env_name).cpu().numpy().flatten()
            return action

    def train_single(
        self,
        env_name,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        update_ac=False,
        update_sy=False,
        extra_args=None,
    ):
        logs = defaultdict(list)


        x, y, u, r, d = replay_buffer.sample(1000)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        N = action.shape[1]

        a = 1 - torch.eye(N, device=device)

        dQ = []
        for i in range(N):
            ai = action * a[i]
            q0, _ = self.critic(state, ai, self.synergy, env_name)
            q1, _ = self.critic(state, action, self.synergy, env_name)
            dQ.append(q1.mean(dim=-1) - q0.mean(dim=-1))
        dQ = torch.stack(dQ, dim=1) # [bs,N]
        self.synergy.update_synergy_weights(env_name, dQ.mean(dim=0).softmax(dim=0).detach().cpu().numpy() * 1.2)

        for it in range(iterations):

            # sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            if update_ac: # test-envs do not update critic
                # select action according to policy and add clipped noise
                with torch.no_grad():
                    noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                    noise = noise.clamp(-noise_clip, noise_clip)
                    next_action = self.actor_target(next_state, self.synergy, env_name=env_name) + noise
                    next_action = next_action.clamp(
                        -self.args.max_action, self.args.max_action
                    )

                    target_Q1, target_Q2 = self.critic_target(next_state, next_action, self.synergy, env_name)
                    target_Q = torch.min(target_Q1.mean(dim=-1, keepdim=True), target_Q2.mean(dim=-1, keepdim=True)) # DDOP
                    target_Q = reward + (done * discount * target_Q)

                # get current Q estimates
                current_Q1, current_Q2 = self.critic(state, action, self.synergy, env_name)

                # compute critic loss
                critic_loss = F.mse_loss(current_Q1.mean(dim=-1, keepdim=True), target_Q) + F.mse_loss(
                    current_Q2.mean(dim=-1, keepdim=True), target_Q) # DDOP
                logs['critic_loss'].append(critic_loss.item())
                logs['target_Q'].append(target_Q.mean().item())

                # optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.args.grad_clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.args.grad_clipping_value
                    )
                self.critic_optimizer.step()

            # delayed policy updates
            if it % policy_freq == 0:

                # compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state, self.synergy, env_name=env_name),
                                             self.synergy, env_name).mean()
                logs['actor_loss'].append(actor_loss.item())

                # optimize the actor
                self.actor_optimizer.zero_grad()
                self.synergy_optimizer.zero_grad()
                actor_loss.backward()

                if self.args.grad_clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.args.grad_clipping_value
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.synergy.parameters(), self.args.grad_clipping_value
                    )
                if update_sy and it % (policy_freq * 2) == 0:
                    self.synergy_optimizer.step()
                if update_ac: # test-envs do not update actors
                    self.actor_optimizer.step()
                    # update the frozen target models
                    for param, target_param in zip(
                        self.critic.parameters(), self.critic_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1 - tau) * target_param.data
                        )

                    for param, target_param in zip(
                        self.actor.parameters(), self.actor_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1 - tau) * target_param.data
                        )

        final_logs = dict()
        for k,v in logs.items():
            final_logs[f"{env_name}_{k}"] = np.mean(v)
        return final_logs

    def train(
        self,
        replay_buffer_list,
        iterations_list,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        graphs=None,
        envs_train_names=None,
        envs_test_names=None,
        extra_args=None,
    ):

        self.models2train()

        per_morph_iter = sum(iterations_list) // len(envs_train_names)
        logs = dict()
        steps = extra_args['steps']
        for env_name in envs_train_names:
            is_test_env = True if env_name in envs_test_names else False
            replay_buffer = replay_buffer_list[env_name]
            self.change_morphology(graphs[env_name])
            single_logs = self.train_single(
                env_name,
                replay_buffer,
                per_morph_iter,
                batch_size=batch_size,
                discount=discount,
                tau=tau,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                policy_freq=policy_freq,
                update_ac = (not is_test_env),
                # update_sy = True,
                update_sy = (not is_test_env), # no adaptation
                extra_args=extra_args,
            )
            logs.update(single_logs)

        extra_return = {'test_status': 0}

        self.models2eval()
        return logs, extra_return


    def get_info(self):
        return self.synergy.get_info()

    def models2eval(self):
        self.actor = self.actor.eval()
        self.actor_target = self.actor_target.eval()
        self.critic = self.critic.eval()
        self.critic_target = self.critic_target.eval()
        self.synergy.eval()

    def models2train(self):
        self.actor = self.actor.train()
        self.actor_target = self.actor_target.train()
        self.critic = self.critic.train()
        self.critic_target = self.critic_target.train()
        self.synergy.train()

    def save(self, fname):
        torch.save(self.actor.state_dict(), "%s_actor.pth" % fname)
        torch.save(self.critic.state_dict(), "%s_critic.pth" % fname)
        torch.save(self.synergy.state_dict(), "%s_synergy.pth" % fname)

    def load(self, fname):
        self.actor.load_state_dict(torch.load("%s_actor.pth" % fname))
        self.critic.load_state_dict(torch.load("%s_critic.pth" % fname))
        self.synergy.load_state_dict(torch.load("%s_synergy.pth" % fname))
