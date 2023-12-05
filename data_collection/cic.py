import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

from data_collection import url_utils
from data_collection.url_utils import DDPGAgent


class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.skill_dim))

        self.pred_net = nn.Sequential(nn.Linear(2 * self.skill_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.skill_dim))

        if project_skill:
            self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Linear(hidden_dim, self.skill_dim))
        else:
            self.skill_net = nn.Identity()  
   
        self.apply(url_utils.weight_init)

    def forward(self, state, next_state, skill):
        assert len(state.size()) == len(next_state.size())
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        query = self.skill_net(skill)
        key = self.pred_net(torch.cat([state, next_state],1))
        return query, key


class CICAgent(DDPGAgent):
    # Contrastive Intrinsic Control (CIC)
    def __init__(self, update_skill_every_step, skill_dim, scale, 
                    project_skill, temp, discount, alpha, 
                    knn_clip, knn_k, knn_avg, knn_rms, **kwargs):
        self.temp = temp
        self.alpha = alpha
        self.discount = discount
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.scale = scale
        self.project_skill = project_skill
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        # create cic first
        self.cic = CIC(self.obs_dim - skill_dim, skill_dim,
                           kwargs['hidden_dim'], project_skill).to(kwargs['device'])

        # particle-based entropy
        rms = url_utils.RMS(self.device)
        self.pbe = url_utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

        # optimizers
        self.cic_optimizer = torch.optim.Adam(self.cic.parameters(),
                                                lr=self.lr)

        self.cic.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.random.uniform(0,1,self.skill_dim).astype(np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, step, time_step):
        if step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def compute_cpc_loss(self, obs, next_obs, skill):
        temperature = self.temp
        eps = 1e-6
        query, key = self.cic.forward(obs, next_obs, skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T) # (b,b))
        
        sim = torch.exp(cov / temperature) 
        neg = sim.sum(dim=-1) # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature) #(b,)
        loss = -torch.log(pos / (neg + eps)) #(b,)
        return loss, cov / temperature

    def update_cic(self, obs, skill, next_obs, step):
        metrics = dict()

        loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
        loss = loss.mean()
        self.cic_optimizer.zero_grad()
        loss.backward()
        self.cic_optimizer.step()

        metrics['cic_loss'] = loss.item()
        metrics['cic_logits'] = logits.norm()

        return metrics

    def compute_intr_reward(self, obs, skill, next_obs, step):
        
        with torch.no_grad():
            loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
      
        reward = loss
        reward = reward.clone().detach().unsqueeze(-1)

        return -reward

    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        reward = self.pbe(source, target) # (b, 1)
        return reward

    def update(self, batch, step):
        metrics = dict()

        obs, action, next_obs, skill = batch

        with torch.no_grad():
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)

        metrics.update(self.update_cic(obs, skill, next_obs, step))

        apt_reward = self.compute_apt_reward(obs, next_obs)
        metrics['apt_reward'] = apt_reward.mean().item()
        cpc_reward = self.compute_intr_reward(obs, skill, next_obs, step)
        intr_reward = self.alpha * apt_reward + (1 - self.alpha) * cpc_reward

        reward = intr_reward * self.scale

        # metrics['extr_reward'] = extr_reward.mean().item()
        metrics['intr_reward'] = intr_reward.mean().item()
        metrics['batch_reward'] = reward.mean().item()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, self.discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, step))

        # update critic target
        url_utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

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
