import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

from data_collection import url_utils


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(url_utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = url_utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(url_utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class DDPGAgent:
    def __init__(self,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 stddev_schedule,
                 stddev_clip,
                 meta_dim=0):

        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.feature_dim = feature_dim
        self.solved_meta = None

        # models
        self.aug = nn.Identity()
        self.encoder = nn.Identity()
        self.obs_dim = obs_shape[0] + meta_dim

        self.actor = Actor(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim).to(device)

        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers

        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)


class CriticSF(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim,
                 sf_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, sf_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(url_utils.weight_init)

    def forward(self, obs, action, task):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        q1 = torch.einsum("bi,bi->b", task, q1).reshape(-1, 1)
        q2 = torch.einsum("bi,bi->b", task, q2).reshape(-1, 1)

        return q1, q2


class APS(nn.Module):
    def __init__(self, obs_dim, sf_dim, hidden_dim):
        super().__init__()
        self.state_feat_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, sf_dim))

        self.apply(url_utils.weight_init)

    def forward(self, obs, norm=True):
        state_feat = self.state_feat_net(obs)
        state_feat = F.normalize(state_feat, dim=-1) if norm else state_feat
        return state_feat


class APSAgent(DDPGAgent):
    def __init__(self, update_task_every_step, sf_dim, knn_rms, knn_k, knn_avg,
                 knn_clip, discount,
                 **kwargs):
        self.sf_dim = sf_dim
        self.update_task_every_step = update_task_every_step
        self.discount = discount

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        # create actor and critic
        super().__init__(**kwargs)

        # overwrite critic with critic sf
        self.critic = CriticSF(self.obs_type, self.obs_dim, self.action_dim,
                               self.feature_dim, self.hidden_dim,
                               self.sf_dim).to(self.device)
        self.critic_target = CriticSF(self.obs_type, self.obs_dim,
                                      self.action_dim, self.feature_dim,
                                      self.hidden_dim,
                                      self.sf_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.lr)

        self.aps = APS(self.obs_dim - self.sf_dim, self.sf_dim,
                       kwargs['hidden_dim']).to(kwargs['device'])

        # particle-based entropy
        rms = url_utils.RMS(self.device)
        self.pbe = url_utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

        # optimizers
        self.aps_opt = torch.optim.Adam(self.aps.parameters(), lr=self.lr)

        self.train()
        self.critic_target.train()

        self.aps.train()

    def get_meta_specs(self):
        return (specs.Array((self.sf_dim,), np.float32, 'task'),)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        task = torch.randn(self.sf_dim)
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def update_aps(self, task, next_obs, step):
        metrics = dict()

        loss = self.compute_aps_loss(next_obs, task)

        self.aps_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.aps_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        metrics['aps_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, task, next_obs, step):
        # maxent reward
        with torch.no_grad():
            rep = self.aps(next_obs, norm=False)
        reward = self.pbe(rep)
        intr_ent_reward = reward.reshape(-1, 1)

        # successor feature reward
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        intr_sf_reward = torch.einsum("bi,bi->b", task, rep).reshape(-1, 1)

        return intr_ent_reward, intr_sf_reward

    def compute_aps_loss(self, next_obs, task):
        """MLE loss"""
        loss = -torch.einsum("bi,bi->b", task, self.aps(next_obs)).mean()
        return loss

    def update(self, batch, step):
        metrics = dict()

        obs_task, action, next_obs_task = batch
        task = obs_task[:, -self.sf_dim:]
        obs = obs_task[:, :-self.sf_dim]
        next_obs = next_obs_task[:, :-self.sf_dim]

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        # freeze successor features at finetuning phase
        metrics.update(self.update_aps(task, next_obs, step))

        with torch.no_grad():
            intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                task, next_obs, step)
            intr_reward = intr_ent_reward + intr_sf_reward

        metrics['intr_reward'] = intr_reward.mean().item()
        metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
        metrics['intr_sf_reward'] = intr_sf_reward.mean().item()

        reward = intr_reward

        # extend observations with task
        obs = torch.cat([obs, task], dim=1)
        next_obs = torch.cat([next_obs, task], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, self.discount,
                               next_obs.detach(), task, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), task, step))

        # update critic target
        url_utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, task,
                      step):
        """diff is critic takes task as input"""
        metrics = dict()

        with torch.no_grad():
            stddev = url_utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action,
                                                      task)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, task)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, task, step):
        """diff is critic takes task as input"""
        metrics = dict()

        stddev = url_utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, task)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        stddev = url_utils.schedule(self.stddev_schedule, step)
        dist = self.actor(h, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]
