import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data import DataLoader, DistributedSampler  
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance
from embeddings_AETR_CLS import TreeAutoencoder,get_expression_cls_embedding,flatten_expr
# from expression_manager import HierarchicalExpression
from typing import Union,Dict,Tuple

# Ensure the custom module "pytrs" is in your PYTHONPATH.
sys.path.append(os.path.abspath("./pytrs"))
from pytrs import (
    parse_sexpr,
    expr_to_str,
    calculate_cost,
    create_rules,
    generate_random_assignments,
    evaluate_expr,Op,Expr,Const,Var,
)



ddp = int(os.environ.get('RANK', -1)) != -1  
deviceids = [0, 1, 2]  

if ddp:  
    assert torch.cuda.is_available(), "DDP requires CUDA"  
    dist.init_process_group(backend='nccl')  
    ddp_rank = int(os.environ['RANK'])  
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  
    ddp_world_size = int(os.environ['WORLD_SIZE'])  
    device = f'cuda:{deviceids[ddp_rank]}'  
    torch.cuda.set_device(device)  
    master_process = ddp_rank == 0  
else:  
    master_process = True  
    ddp_rank = 0  
    ddp_world_size = 1  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set number of threads and create the rules dictionary.
torch.set_num_threads(4)
rules = create_rules(vector_size=8)

embeddings_model = TreeAutoencoder()  
state_dict = torch.load("./trained_models/embeddings_ROT_15_32_5m_10742576.pth", map_location=device,weights_only=True)
new_sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}
embeddings_model.load_state_dict(new_sd)
embeddings_model.to(device) 
embeddings_model.eval()
print("loaded")
############################################
# Custom Feature Extractor                 #
############################################
class CustomFeaturesExtractor(nn.Module):
    """
    A minimal custom feature extractor that takes the environment's
    numeric embedding (obs["observation"]) and maybe does
    a small transformation. If you really want a no-op, just skip the net.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()
        # The environment returns shape (embed_dim,) for obs["observation"].
        # Let's retrieve that embed_dim:
        self._embed_dim = observation_space["observation"].shape[0]
        self._features_dim = features_dim
        embed_dim = self._embed_dim
        # self.net = nn.Sequential(
        #     nn.Linear(embed_dim, 512),
        #     nn.LayerNorm(512),
        #     nn.GELU(),
            
        #     nn.Linear(512, 512),
        #     nn.LayerNorm(512),
        #     nn.GELU(),
            
        #     nn.Linear(512, features_dim),
        #     nn.LayerNorm(features_dim),
        #     nn.GELU(),
            
        #     nn.Linear(features_dim, features_dim),
        #     nn.ReLU(),
        # )
        # self.net = nn.Sequential(
        #     nn.Linear(embed_dim, 256),
        #     nn.LayerNorm(256),
        #     nn.GELU(),

        #     nn.Linear(256, features_dim),
        #     nn.ReLU(),
        # )

    def forward(self, obs_dict: dict) -> torch.Tensor:
        x = obs_dict["observation"]
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.net.parameters()).device)
        #return self.net(x)
        return x

    @property
    def features_dim(self):
        # For stable-baselines3, we usually define a features_dim
        return self._features_dim

############################################
# Hierarchical Maskable Policy             #
############################################
class HierarchicalMaskablePolicy(nn.Module):
    def __init__(self, observation_space, action_space, lr_schedule, **policy_kwargs):
        """
        Custom hierarchical policy that selects a high-level rule and then a sub-action (position).
        Expected extra parameters (via policy_kwargs):
          - rule_dim: Total number of high-level rules.
          - max_positions: Number of positions (sub-actions) available per rule.
          - features_dim: (Optional) Dimension of feature extractor output (default: 256).
          - lr: Learning rate for the separate optimizers (default: 3e-4).
          - ent_coef: Entropy coefficient (default: 0.01).
        """
        super().__init__()
        # Extract extra parameters with defaults.
        self.rule_dim = policy_kwargs.pop("rule_dim", 5)
        self.max_positions = policy_kwargs.pop("max_positions", 2)
        features_dim = policy_kwargs.pop("features_dim", 256)
        lr = policy_kwargs.pop("lr", 3e-4)
        self.ent_coef = policy_kwargs.pop("ent_coef", 0.5)
        self.rule_weight = policy_kwargs.pop("rule_weight", 1.0)
        self.pos_weight  = policy_kwargs.pop("pos_weight", 5.0)
        # Feature extractor.
        self.encoder = CustomFeaturesExtractor(observation_space, features_dim)

        # Rule selection network.
        # self.rule_head = nn.Sequential(
        #     nn.Linear(features_dim, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.rule_dim),
        # )
        self.rule_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.rule_dim),
        )
        # Position selection network.
        # self.pos_head = nn.Sequential(
        #     nn.Linear(features_dim + self.rule_dim, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.max_positions),
        # )
        self.pos_head = nn.Sequential(
            nn.Linear(features_dim + self.rule_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.max_positions),
        )
        # Value network.
        # self.value_net = nn.Sequential(
        #     nn.Linear(features_dim, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),

        #     nn.Linear(128, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),

        #     nn.Linear(64, 1),
        # )
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )
        actor_params  = (
            list(self.encoder.parameters())
            + list(self.rule_head.parameters())
            + list(self.pos_head.parameters())
        )
        critic_params = self.value_net.parameters()
        # Separate optimizers.
        # self.rule_optimizer = optim.Adam(self.rule_net.parameters(), lr=lr)
        # self.position_optimizer = optim.Adam(self.position_net.parameters(), lr=lr)
        #self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-4)
        #self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_params,  "lr": lr},
                {"params": critic_params, "lr": lr * 0.1},
            ]
        )
    def _rule_dist(self, enc: torch.Tensor, rule_mask: torch.Tensor) -> Categorical:
        logits = self.rule_head(enc)
        logits = torch.where(rule_mask, logits, torch.tensor(-torch.inf, device=enc.device))
        return Categorical(logits=logits)

    def _pos_dist(
        self, enc: torch.Tensor, one_hot_rule: torch.Tensor, pos_mask: torch.Tensor
    ) -> Categorical:
        logits = self.pos_head(torch.cat([enc, one_hot_rule], dim=1))
        logits = torch.where(pos_mask, logits, torch.tensor(-torch.inf, device=enc.device))
        return Categorical(logits=logits)
    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        """Minimal imitation of ActorCriticPolicy.predict()."""
        device = next(self.parameters()).device
        if isinstance(observation, dict):
            obs = {k: torch.as_tensor(v, device=device) for k, v in observation.items()}
        else:
            obs = torch.as_tensor(observation, device=device)
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions.cpu().numpy(), state

    # def predict(self, observation, state=None, episode_start=None, deterministic=False):
    #     device = next(self.parameters()).device
    #     # If observation is a dict, ensure each entry has a batch dimension.
    #     if isinstance(observation, dict):
    #         new_obs = {}
    #         for key, value in observation.items():
    #             # If it's a NumPy array and 1D, add a batch dimension.
    #             if isinstance(value, np.ndarray):
    #                 if value.ndim == 1:
    #                     value = np.expand_dims(value, axis=0)
    #                 new_obs[key] = torch.as_tensor(value, device=device)
    #             # If it's a torch tensor and 1D, unsqueeze it.
    #             elif torch.is_tensor(value):
    #                 if value.dim() == 1:
    #                     value = value.unsqueeze(0)
    #                 new_obs[key] = value.to(device)
    #             else:
    #                 new_obs[key] = value
    #         observation = new_obs
    #     else:
    #         if isinstance(observation, np.ndarray) and observation.ndim == 1:
    #             observation = np.expand_dims(observation, axis=0)
    #         observation = torch.as_tensor(observation, device=device)
        
    #     # Now call forward with the properly batched observation.
    #     action, _, _ = self.forward(observation, deterministic=deterministic)
    #     return action.cpu().numpy(), state
    
        

    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (actions, values, log_prob)."""
        enc = self.encoder(obs)

        # action_mask comes in flat shape (B, rule_dim*max_pos)
        mask = obs["action_mask"].bool()
        B = mask.size(0)
        mask = mask.view(B, self.rule_dim, self.max_positions)

        # stage 1: rule
        rule_mask = mask.any(dim=2)  # (B, rule_dim)
        rule_dist = self._rule_dist(enc, rule_mask)
        rule_action = rule_dist.mode if deterministic else rule_dist.sample()

        # stage 2: pos given rule
        one_hot_rule = torch.nn.functional.one_hot(rule_action, self.rule_dim).float()
        pos_mask = mask[torch.arange(B, device=enc.device), rule_action]  # (B,max_pos)
        pos_dist = self._pos_dist(enc, one_hot_rule, pos_mask)
        pos_action = pos_dist.mode if deterministic else pos_dist.sample()

        # combine
        flat_action = rule_action * self.max_positions + pos_action
        log_prob = rule_dist.log_prob(rule_action) + pos_dist.log_prob(pos_action)

        # critic
        value = self.value_net(enc).squeeze(-1)
        return flat_action, value, log_prob

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used inside PPO.update; returns (values, log_prob, entropy)."""
        enc = self.encoder(obs)
        B = actions.size(0)
        mask = obs["action_mask"].bool().view(B, self.rule_dim, self.max_positions)

        # decompose flat → (rule,pos)
        rule_actions = (actions // self.max_positions).long()
        pos_actions = (actions % self.max_positions).long()

        rule_mask = mask.any(dim=2)
        rule_dist = self._rule_dist(enc, rule_mask)
        one_hot_rule = torch.nn.functional.one_hot(rule_actions, self.rule_dim).float()

        pos_mask = mask[torch.arange(B, device=enc.device), rule_actions]
        pos_dist = self._pos_dist(enc, one_hot_rule, pos_mask)

        log_prob = rule_dist.log_prob(rule_actions) + pos_dist.log_prob(pos_actions)
        entropy = rule_dist.entropy() + pos_dist.entropy()
        value = self.value_net(enc).squeeze(-1)
        return value, log_prob, entropy
    
    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc = self.encoder(obs)
        return self.value_net(enc).squeeze(-1)
    # def predict_values(self, obs, action_masks=None):
    #     """
    #     Returns the value estimates for the given observations.
    #     """
    #     if action_masks is None:
    #         if isinstance(obs, dict) and "action_mask" in obs:
    #             action_masks = obs["action_mask"]
    #         else:
    #             raise ValueError("Action masks must be provided!")
    #     features = self.features_extractor(obs)
    #     return self.value_net(features)

    def set_training_mode(self, mode: bool):
        self.train(mode)

############################################
# Custom PPO Implementation (HierarchicalPPO)#
############################################
class HierarchicalPPO(PPO):
    # def __init__(self, *args, rule_dim=5, max_positions=32, **kwargs):
    #     """
    #     rule_dim: Total number of high-level rules.
    #     max_positions: Number of sub-actions (positions) available per rule.
    #     """
    #     if "policy_kwargs" not in kwargs:
    #         kwargs["policy_kwargs"] = {}
    #     kwargs["policy_kwargs"]["rule_dim"] = rule_dim
    #     kwargs["policy_kwargs"]["max_positions"] = max_positions
    #     super().__init__(*args, **kwargs)
    #     self.ep_info_buffer = []

    


    def _update_info_buffer(self, infos, dones=None):  # type: ignore[override]
        # Delegate completely to the parent implementation.
        return super()._update_info_buffer(infos, dones)

    # def obs_to_tensor(self,obs, device):
    #     """
    #     Convert observation to PyTorch tensor.
    #     """
    #     if isinstance(obs, dict):
    #         return {
    #             key: torch.as_tensor(value).to(device)
    #             for key, value in obs.items()
    #         }
    #     return torch.as_tensor(obs).to(device)
    # def collect_rollouts(
    #     self,
    #     env,
    #     callback,
    #     rollout_buffer,
    #     n_rollout_steps: int,
    # ) -> bool:
    #     print("called")
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     n_steps = 0
    #     rollout_buffer.reset()

    #     # Initialize episode info buffer for this collection phase
    #     self.ep_info_buffer = []
    #     print("Cleared ep_info_buffer")  # Debug print

    #     callback.on_rollout_start()
    #     while n_steps < n_rollout_steps:
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             self.policy.reset_noise(env.num_envs)

    #         with torch.no_grad():
    #             obs_tensor = self.obs_to_tensor(self._last_obs, self.device)
    #             actions, values, log_probs = self.policy.forward(obs_tensor)
    #         actions = actions.cpu().numpy()

    #         # Rescale and perform action
    #         clipped_actions = actions
    #         if isinstance(self.action_space, gym.spaces.Box):
    #             clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

    #         step_result = env.step(clipped_actions)
    #         if len(step_result) == 4:
    #             new_obs, rewards, dones, infos = step_result
    #             truncated = dones
    #         else:
    #             new_obs, rewards, dones, truncated, infos = step_result

    #         self.num_timesteps += env.num_envs

    #         # Store episode info
    #         for info in infos:
    #             if "episode" in info:
    #                 #print(f"Found episode info: {info['episode']}")  # Debug print
    #                 self.ep_info_buffer.append(info["episode"])
    #                 #print(f"Current buffer size: {len(self.ep_info_buffer)}")  # Debug print


    #         callback.update_locals(locals())
    #         if callback.on_step() is False:
    #             return False

    #         n_steps += 1

    #         if isinstance(self.action_space, gym.spaces.Discrete):
    #             actions = actions.reshape(-1, 1)

    #         # Handle timeout by bootstrapping with value function
    #         for idx, done in enumerate(dones):
    #             if (
    #                 done
    #                 and infos[idx].get("terminal_observation") is not None
    #                 and infos[idx].get("TimeLimit.truncated", False)
    #             ):
    #                 terminal_obs = self.obs_to_tensor(infos[idx]["terminal_observation"], self.device)
    #                 with torch.no_grad():
    #                     terminal_value = self.policy.predict_values(terminal_obs)[0]
    #                 rewards[idx] += self.gamma * terminal_value

    #         rollout_buffer.add(
    #             self._last_obs,
    #             actions,
    #             rewards,
    #             self._last_episode_starts,
    #             values,
    #             log_probs,
    #         )
    #         self._last_obs = new_obs
    #         self._last_episode_starts = dones

    #     with torch.no_grad():
    #         obs_tensor = self.obs_to_tensor(new_obs, self.device)
    #         values = self.policy.predict_values(obs_tensor)

    #     rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    #     print(f"End of collection. Buffer size: {len(self.ep_info_buffer)}")  # Debug print
    #     callback.on_rollout_end()

    #     return True

    # def train(self) -> None:
    #     """
    #     One complete PPO update.  We iterate over *all* epochs and
    #     *all* mini‑batches, computing a **single scalar loss**
    #     that combines policy/value/entropy terms.  We then call
    #     `.zero_grad()`, `loss.backward()`, `clip_grad_norm_()`,
    #     and finally `.step()` on the three optimizers.
    #     """
    #     #self.policy.ent_coef = max(0.01, self.policy.ent_coef * 0.99)

    #     self.rollout_buffer.advantages = (self.rollout_buffer.advantages -
    #                                       self.rollout_buffer.advantages.mean()) / (
    #                                           self.rollout_buffer.advantages.std() + 1e-8
    #                                       )

    #     logger = self.logger
    #     policy = self.policy
    #     rb = self.rollout_buffer

    #     last_policy_loss = last_value_loss = last_entropy_loss = 0.0
    #     old_values_for_ev, returns_for_ev = None, None

    #     for _ in range(self.n_epochs):
    #         for rollout_data in rb.get(batch_size=self.batch_size):

                
    #             if isinstance(rollout_data.observations, dict):
    #                 obs_dict = {
    #                     "observation": rollout_data.observations["observation"],
    #                     "action_mask": rollout_data.observations["action_mask"],
    #                 }
    #                 action_masks = rollout_data.observations["action_mask"]
    #             else:
    #                 obs_dict = rollout_data.observations
    #                 action_masks = None

    #             actions     = rollout_data.actions
    #             old_logprob = rollout_data.old_log_prob
    #             old_values  = rollout_data.old_values
    #             returns     = rollout_data.returns
    #             adv         = rollout_data.advantages

    #             old_values_for_ev, returns_for_ev = old_values, returns

               
    #             values, log_prob, entropy = policy.evaluate_actions(
    #                 obs_dict, actions, action_masks=action_masks
    #             )

                
    #             ratio = torch.exp(log_prob - old_logprob)
    #             clip_range = (
    #             self.clip_range(self._current_progress_remaining)
    #             if callable(self.clip_range)
    #             else self.clip_range
    #             )
    #             pg_loss = -torch.min(
    #                 ratio * adv,
    #                 torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv,
    #             ).mean()

    #             # value_loss = 0.5 * (returns - values).pow(2).mean()
    #             entropy_loss = -entropy.mean()
    #             clip_range_vf = (
    #             self.clip_range_vf(self._current_progress_remaining)
    #             if callable(self.clip_range_vf)
    #             else self.clip_range_vf
    #             )

    #         # 2) Unclipped squared error
    #             vf_unclipped = (values - returns).pow(2)

    #         # 3) Clip value predictions around old_values
    #             values_clipped = old_values + torch.clamp(
    #             values - old_values,
    #             -clip_range_vf,
    #             clip_range_vf
    #             )
    #             vf_clipped = (values_clipped - returns).pow(2)

    #         # 4) Take the maximum per sample (PPO’s recommendation) and mean
    #             value_loss = 0.5 * torch.max(vf_unclipped, vf_clipped).mean()

    #             loss = pg_loss + self.vf_coef * value_loss + policy.ent_coef * entropy_loss

                
    #             # policy.rule_optimizer.zero_grad()
    #             # policy.position_optimizer.zero_grad()
    #             policy.optimizer.zero_grad()
    #             #policy.value_optimizer.zero_grad()

    #             loss.backward()
    #             nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)

    #             # policy.rule_optimizer.step()
    #             # policy.position_optimizer.step()
    #             policy.optimizer.step()
    #             #policy.value_optimizer.step()

    #             last_policy_loss  = pg_loss.detach()
    #             last_value_loss   = value_loss.detach()
    #             last_entropy_loss = entropy_loss.detach()

       
    #     if old_values_for_ev is not None:
    #         ev = explained_variance(
    #             old_values_for_ev.cpu().numpy().flatten(),
    #             returns_for_ev.cpu().numpy().flatten(),
    #         )
    #         logger.record("train/explained_variance", ev)

    #     logger.record("train/policy_loss",  last_policy_loss.item())
    #     logger.record("train/value_loss",   last_value_loss.item())
    #     logger.record("train/entropy_loss", last_entropy_loss.item())
    #     logger.record("train/entropy_coef", policy.ent_coef)


############################################
# Custom Environment (fheEnv)              #
############################################
class fheEnv(gym.Env):
    def __init__(self, rules_list, expressions, max_positions=2):
        """
        rules_list: List of rule names (strings); one should be "END".
        expressions: List of expression strings.
        max_positions: Maximum number of valid positions per rule.
        """
        super().__init__()
        self.rules = rules_list
        self.expressions = expressions
        self.max_positions = max_positions
        self.max_steps =    60
        self.max_expression_size = 10000
        self.initial_cost = 0
        self.embedding_dim = 256
        self.initial_vectorization_potential = 0
        self.vectorizations_applied = 0
        self.vectorization_helper = 0
        # Action space: each rule has max_positions sub-actions.
        self.action_space = spaces.Discrete(len(self.rules) * self.max_positions)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32
            ),
            "action_mask": spaces.Box(0, 1, (len(self.rules) * self.max_positions,), np.float32)
        })
        self.reset()

    def get_cost(self, expr: str) -> float:
        return calculate_cost(parse_sexpr(expr))
    def _embed_expression(self, expr: str) -> np.ndarray:
        expr_tree = parse_sexpr(expr)
        with torch.no_grad():
            emb = get_expression_cls_embedding(expr_tree, embeddings_model)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)
    
    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.rules) * self.max_positions, dtype=np.float32)
        parsed = parse_sexpr(self.expression)
        for rule_idx, rule_name in enumerate(self.rules):
            if rule_name == "END":
                mask[rule_idx * self.max_positions] = 1.0
                continue
            rule_obj = rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            valid_positions = min(len(matches), self.max_positions)
            if valid_positions > 0:
                start = rule_idx * self.max_positions
                mask[start:start + valid_positions] = 1.0
        return mask

    def _encode_expression(self, expr: str) -> np.ndarray:
        encoded = np.zeros(self.max_expression_size, dtype=np.uint8)
        bytes_data = expr.encode("ascii", "ignore")[:self.max_expression_size]
        encoded[:len(bytes_data)] = np.frombuffer(bytes_data, dtype=np.uint8)
        return encoded
    def vectorisation_potential(self, expr: str) -> float:
        simple_vectorisation_potential = 0
        parsed = parse_sexpr(expr)
        simple_vectorisation_potentials = {"4":[],"8":[],"16":[],"32":[]}
        for vector_size in [4,8,16,32]:
            for name in ["add-vectorize-","neg-vectorize-","sub-vectorize-","mul-vectorize-"]:
                rule_obj = rules[name+str(vector_size)]
                new_potential_array = rule_obj.coverage_progress(parsed)
                for i,potential in enumerate(new_potential_array):
                    if len(simple_vectorisation_potentials[str(vector_size)]) <= i:
                        simple_vectorisation_potentials[str(vector_size)].append(0)
                    value,vec_size,special = potential
                    if (value - special) > 1:
                        simple_vectorisation_potentials[str(vector_size)][i] = max(simple_vectorisation_potentials[str(vector_size)][i],value/vec_size)
        return sum(sum(vals) for vals in simple_vectorisation_potentials.values())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the pointer if it doesn't exist
        if not hasattr(self, "current_index"):
            self.current_index = 0

        # Use the next expression in order
        #self.expression = self.expressions[self.current_index]
        self.expression = random.choice(self.expressions)
        # Increment and wrap the pointer
        self.vectorizations_applied = 0
        self.vectorization_helper = 0
        self.current_index = (self.current_index + 1) % len(self.expressions)
        self.initial_vectorization_potential = 0
        self.initial_expression = self.expression
        #self.he = HierarchicalExpression(self.expression, min_depth=1000)
        #self.abstracted_expression = self.he.get_abstracted()
        self.initial_vectorization_potential = self.vectorisation_potential(self.expression)
        #self.sub_vecs = self.he.collect_all_vector_subexpr_strings()
        #self.current_abstracted_index = 0
        self.steps = 0
        self.initial_cost = self.current_cost = self.get_cost(self.expression)
        return {
            "observation": self._embed_expression(self.expression),
            "action_mask": self.get_action_mask()
        }, {}


    def step(self, action: int):
        self.steps += 1
        rule_idx = action // self.max_positions
        pos_idx = action % self.max_positions
        rule_name = self.rules[rule_idx]
        terminated = False
        truncated = False
        reward = 0
        old_vector_pot = self.vectorisation_potential(self.expression)
        # print("Old expression : ",self.expression)
        # #print("Old abstracted expression : ",self.abstracted_expression)
        # print("Old Cost : ",self.current_cost)
        # print("Old Vectorisation Potential :", old_vector_pot )
        vectorisation_potential = 0

        if rule_name == "END":
            # reward = ( (self.initial_cost - self.current_cost) / self.initial_cost * 10  )
            # isValid = self._valid_end_action(self.expression)
            # if not isValid:
            #     reward = -10
            # else:
            #     terminated = True
            #     truncated = True
            #     if self.initial_vectorization_potential > old_vector_pot:
            #         if self.vectorizations_applied == 0:
            #             reward = -10
            #         else:  
            #             reward += self.vectorizations_applied * 5
            #     else:
            #         if  old_vector_pot > self.initial_vectorization_potential:
            #             if self.initial_vectorization_potential == 0 and old_vector_pot != 0:
            #                 if self.vectorizations_applied == 0:
            #                     reward = -10
            #                 else:  
            #                     reward += self.vectorizations_applied * 2 
            #             else:
            #                 reward += (old_vector_pot - self.initial_vectorization_potential)*50 + self.vectorizations_applied * 2 
                    # else:
                    #     reward -= old_vector_pot*10
            #reward += self.vectorizations_applied * 2
            # isValid = self._valid_end_action(self.abstracted_expression)
            # if(not isValid):
            #     reward = -10
            # else:
            #     reward = 10
            isValid = self._valid_end_action(self.expression)
            if not isValid:
                reward = -10
                reward += self.vectorization_helper * 5 + self.vectorizations_applied * 5
                terminated = True
                truncated = False
            else:
                terminated = True
                truncated = False
                reward = 10
        else:
            #expression_abstracted = self.he.get_abstracted()
            parsed = parse_sexpr(self.expression)
            rule_obj = rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            # if pos_idx >= len(matches):
            #    reward = -10
            # else:
            k, _ = matches[pos_idx]
            new_expr_tree = rule_obj.apply_rule(parsed, path=k)
            temp = expr_to_str(new_expr_tree)
            self.expression = temp
            #self.he = HierarchicalExpression(self.expression, min_depth=10000000)
            #self.abstracted_expression = self.he.get_abstracted()
            #self.sub_vecs = self.he.collect_all_vector_subexpr_strings()
            new_cost = self.get_cost(self.expression)
            reward =( ( self.current_cost - new_cost) / self.current_cost ) * 100
            
            vectorisation_potential = self.vectorisation_potential(self.expression)
            if not rule_name.startswith((
                    "add-vectorize", "mul-vectorize", "neg-vectorize",
                    "sub-vectorize", "rot-add-vectorize", "rot-min-vectorize",
                    "rot-mul-vectorize"
                )):
                if reward < 0:
                    reward = -2
                if reward > 0:
                    reward = 1
                if vectorisation_potential - old_vector_pot > 0:
                    reward = 2
                    self.vectorization_helper += 1
                if vectorisation_potential - old_vector_pot < 0:
                    reward = -4
                    self.vectorization_helper -= 1
            else:
                self.vectorizations_applied += 1
                self.initial_vectorization_potential = vectorisation_potential
                
            self.current_cost = new_cost               
            if (self.steps >= self.max_steps):
                terminated = True
                    #reward = ( (self.initial_cost - self.current_cost) / self.initial_cost * 100  )
        # if reward < 0 and reward > -2:
        #     reward = -2
        info = {"expression": self.expression}
        # print("New expression : ",self.expression)
        # print("New Cost : ",self.current_cost)
        # print("New Vectorisation Potential :", vectorisation_potential)
        # print("Reward : ",reward)
        # print("Rule Name : ",rule_name)
        # print("At Position : ",pos_idx)
        embedding = self._embed_expression(self.expression)
        if embedding is None:
            terminated = True
            truncated = True
            #reward = ( (self.initial_cost - self.current_cost) / self.initial_cost * 100  )
        else:
            terminated = terminated or (self.steps >= self.max_steps)
        if terminated or truncated:
            info["episode"] = {
                "r": reward,
                "l": self.steps,
                "t": None
            }   
        
        
        return {
            "observation": embedding,
            "action_mask": self.get_action_mask()
        }, reward, terminated, truncated, {"expression": self.expression}

    def _valid_end_action(self,expr: str) -> bool:
        expr_tree = parse_sexpr(expr)
        vectorization_potenial = self.vectorisation_potential(expr)
        action_mask = self.get_action_mask()
        isValid = True
        for i, rule_name in enumerate(self.rules):
            if rule_name == "END":
                continue
            rule_obj = rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(expr_tree)
            if len(matches) > 0:
                for i,match in enumerate( matches):
                    if i >= self.max_positions:
                        break
                    k, _ = match
                    new_expr_tree = rule_obj.apply_rule(expr_tree, path=k)
                    temp = expr_to_str(new_expr_tree)
                    if calculate_cost(new_expr_tree) < self.current_cost:
                        isValid = False
                        break
                    if self.vectorisation_potential(temp) > vectorization_potenial:
                        isValid = False
                        break
            if not isValid:
                break
        return isValid
    

############################################
# Helper Functions & Main Execution        #
############################################
def get_token_sequence(exp_str: str):
    """
    Parses the expression string, flattens it, and returns a tuple of node IDs.
    """
    expr = parse_sexpr(exp_str)
    flat = flatten_expr(expr)
    node_ids = tuple(entry["node_id"] for entry in flat)
    return node_ids
import re

def calculate_vector_size(expression):
    if expression.startswith("(Vec ") and expression.endswith(")"):
        content = expression[5:-1].strip()
    else:
        return 0 
    
    element_count = 0
    paren_depth = 0
    current_element = ""
    
    for char in content:
        if char == '(' or char == '[':
            paren_depth += 1
            current_element += char
        elif char == ')' or char == ']':
            paren_depth -= 1
            current_element += char
        elif char == ' ' and paren_depth == 0:
            if current_element:
                element_count += 1
                current_element = ""
        else:
            current_element += char
    
    if current_element:
        element_count += 1
    
    return element_count

def load_expressions(file_path: str,validation_exprs = []):
    # Precompute token sequences for the validation expressions.
    
    validation_token_set = set()
    for val in validation_exprs:
        # Extract the expression part (before the "|||") if needed.
        exp_str = val.split("|||")[0].strip() if "|||" in val else val.strip()
        try:
            token_seq = get_token_sequence(exp_str)
            validation_token_set.add(token_seq)
        except Exception as e:
            print("Error processing validation expression:", exp_str, e)
    
    unique_expressions = {}
    recap = { "1": 0, "4": 0, "8": 0, "16": 0, "32": 0 }

    with open(file_path, "r") as f:
        for line in f:
            exp_str = line.split("|||")[0].strip()
            if not exp_str:
                continue
            try:
                expr = parse_sexpr(exp_str)
                if not expr.validate_expression():
                    #print("Invalid: ", exp_str)
                    continue
                vec_size = len(expr.args)
                if  not (vec_size > 0 and (vec_size & (vec_size - 1) == 0) and vec_size <= 32):
                    continue
                token_seq = get_token_sequence(exp_str)
                # Skip expression if its token sequence is in the validation set.
                if token_seq in validation_token_set:
                    continue

                # Only add if token sequence hasn't been seen yet.
                if token_seq not in unique_expressions:
                    recap[str(vec_size)] += 1
                    unique_expressions[token_seq] = exp_str
            except Exception as e:
                pass
                #print("Error parsing expression:", exp_str, e)
    print(recap)
    print("Number of unique valid expressions (excluding validation):", len(unique_expressions))
    return list(unique_expressions.values())
   
from agent_logger import log_training_details
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

def linear_schedule(initial: float, final: float):
    return lambda p: final + (initial - final) * p
from stable_baselines3.common.monitor import Monitor
def train_agent(expressions_file: str, total_timesteps: int = 1000000, num_envs: int = 8):
    validation_exprs = load_expressions("test_expressions_merged.txt")
    expressions = load_expressions(expressions_file,validation_exprs)
    max_positions = 32
    rules_list = list(rules.keys()) + ["END"]
    env_fns = [lambda: Monitor(fheEnv(rules_list, expressions, max_positions=max_positions))
           for _ in range(num_envs)]
    env = SubprocVecEnv(env_fns)
    
    #env = fheEnv(rules_list, expressions, max_positions=max_positions)
    lr_schedule = linear_schedule(1e-4, 1e-4)
    train_env = VecNormalize(
    env,
    norm_obs=False,
    norm_reward=True,
    clip_reward=20.0 
    )
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    run_name = f"model_{job_id}"
    
    tensorboard_log_dir = f"./tensorboard/{run_name}"
    
    model_params = {
        "policy": HierarchicalMaskablePolicy,
        "env": env,
        "learning_rate": lr_schedule,
         "n_steps":256,
        "batch_size":64,
        "gamma":0.99,
        "gae_lambda":0.95,
        #"clip_range_vf":0.2,
        "n_epochs":15,
        "verbose":1,
        "ent_coef":0.1,
        "verbose": 1,
        "tensorboard_log": tensorboard_log_dir,
        "policy_kwargs": {
            "rule_dim": len(rules_list),
            "max_positions": max_positions,
            "rule_weight": 1.0,    # default
            "pos_weight": 5.0, 
            "ent_coef": 0.1,
        },
    }
    
    log_training_details(model_params, job_id, num_data=len(expressions), 
                         num_actions=len(rules_list), total_timesteps=total_timesteps, 
                         output_model_name=run_name, notes="2 level hierarchical PPO max steps 60 and 8 envs")
    
    model = HierarchicalPPO(
        policy=HierarchicalMaskablePolicy,
        env=train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        verbose=1,
        ent_coef=0.01,
        #vf_coef=0.5,
        #max_grad_norm=0.5,
        tensorboard_log=tensorboard_log_dir,
        # rule_dim=len(rules_list),
        # max_positions=max_positions,
        policy_kwargs={
            "rule_dim": len(rules_list),
            "max_positions": max_positions,
            "ent_coef": 0.01,
             "rule_weight": 1.0,
            "pos_weight": 1.0,
        },
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=1)
    model.save(run_name)
    print(f"Model saved as {run_name}")
    return model


from agent_logger import log_test_results

def test_agent(expressions_file: str, model_filepath: str):
    expressions = load_expressions(expressions_file)
    rules_list = list(rules.keys()) + ["END"]
    results = []
    max_positions = 32

    #env = fheEnv(rules_list, expressions, max_positions=max_positions)
    env = DummyVecEnv([
    lambda: Monitor(fheEnv(rules_list, expressions, max_positions=max_positions))
    ])
    test_env = VecNormalize(
    env,
    norm_obs=False,
    norm_reward=True,
    clip_reward=20.0 
    )
    model = HierarchicalPPO(
        policy=HierarchicalMaskablePolicy,
        env=test_env,
        #rule_dim=len(rules_list),
        #max_positions=max_positions,
        policy_kwargs={
            "rule_dim": len(rules_list),
            "max_positions": max_positions,
            "ent_coef": 0.1,
        },
    )
    model = model.load(model_filepath)
    def predict_method(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        device = next(self.parameters()).device
        if isinstance(observation, dict):
            obs = {k: torch.as_tensor(v, device=device) for k, v in observation.items()}
        else:
            obs = torch.as_tensor(observation, device=device)
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions.cpu().numpy(), state
    import types
    model.policy.predict = types.MethodType(predict_method, model.policy)
    num_expr = len(expressions)
    for _ in range(num_expr):
        # 1) reset and grab initials
        obs = env.reset()
        wrapper  = env.envs[0]       # Monitor
        fhe_env   = wrapper.env     # your FHE env
        test_expr = fhe_env.initial_expression
        initial_cost = fhe_env.initial_cost

        done = False
        steps = 0

        # placeholders for the “last” pre-step snapshot
        last_expr = None
        last_cost = None

        while not done:
            # snapshot *before* stepping
            last_expr = fhe_env.expression
            last_cost = fhe_env.current_cost

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            done = bool(dones[0])
            steps += 1

        # at this point `fhe_env` has already auto-reset,
        # but our `last_*` hold the true final values.
        results.append({
            "Test Expression":   test_expr,
            "Final Expression":  last_expr,
            "Initial Cost":      initial_cost,
            "Final Cost":        last_cost,
            "Steps":             steps
        })
    
        print({
            "Test Expression":   test_expr,
            "Final Expression":  last_expr,
            "Initial Cost":      initial_cost,
            "Final Cost":        last_cost,
            "Steps":             steps
        })
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    sheet_name = f"HierarchicalPPO_Test_{job_id}"
    log_test_results(results, sheet_name=sheet_name)
    
    return results

def run_agent(expressions_file: str, model_filepath: str,output_file: str):
    expressions = load_expressions(expressions_file)
    if not len(expressions):
        print("No valid expressions found in the file.")
        sys.exit(1)
        return
    rules_list = list(rules.keys()) + ["END"]
    results = []
    max_positions = 32

    #env = fheEnv(rules_list, expressions, max_positions=max_positions)
    env = DummyVecEnv([
    lambda: Monitor(fheEnv(rules_list, expressions, max_positions=max_positions))
    ])
    test_env = VecNormalize(
    env,
    norm_obs=False,
    norm_reward=True,
    clip_reward=20.0 
    )
    model = HierarchicalPPO(
        policy=HierarchicalMaskablePolicy,
        env=test_env,
        #rule_dim=len(rules_list),
        #max_positions=max_positions,
        policy_kwargs={
            "rule_dim": len(rules_list),
            "max_positions": max_positions,
            "ent_coef": 0.1,
        },
    )
    model = model.load(model_filepath)
    def predict_method(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        device = next(self.parameters()).device
        if isinstance(observation, dict):
            obs = {k: torch.as_tensor(v, device=device) for k, v in observation.items()}
        else:
            obs = torch.as_tensor(observation, device=device)
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions.cpu().numpy(), state
    import types
    model.policy.predict = types.MethodType(predict_method, model.policy)

    obs = env.reset()
    wrapper  = env.envs[0]
    fhe_env   = wrapper.env
    test_expr = fhe_env.initial_expression
    initial_cost = fhe_env.initial_cost

    done = False
    steps = 0

    last_expr = None
    last_cost = None
    final_expr = None
    while not done:
        last_expr = fhe_env.expression
        last_cost = fhe_env.current_cost

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if rewards[0] < 0 and not final_expr:
            final_expr = last_expr
        if rewards[0] > 0 and final_expr:
            final_expr = None
        if rewards[0] < 0 and final_expr:
            final_expr = last_expr
            break
        done = bool(dones[0])
        steps += 1
    if not final_expr:
        final_expr = last_expr
    parsed = parse_sexpr(final_expr)
    vec_sizes=" ".join(str(x) for x in calc_vec_sizes(parsed))
    print(vec_sizes)
    with open (output_file, "w") as f:
        f.write(last_expr+"\n"+vec_sizes)

def calc_vec_sizes(expr:Expr):
    vec_sizes=[]
    def rec(node:Expr):
        if isinstance(node, (Const, Var)):
            return
        if isinstance(node, Op):
            if node.op == "Vec":
                vec_sizes.append(len(node.args))
            else:
                for arg in node.args:
                    rec(arg)
            
    rec(expr)
    return vec_sizes
    
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test] [model_filepath (for test)]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    if mode == "train":
        train_agent("cleaned_dataset.txt", total_timesteps=2000000)
    elif mode == "test":
        if len(sys.argv) < 3:
            print("Usage: python main.py test [model_filepath]")
            sys.exit(1)
        model_filepath = sys.argv[2]
        test_agent("test_expressions_merged.txt", model_filepath)
    elif mode == "run":
        if len(sys.argv) < 5:
            print("Usage: python main.py run [model_filepath] [input_file] [output_file]")
            sys.exit(1)
        model_filepath = sys.argv[2]
        input_file = sys.argv[3]
        output_file = sys.argv[4]
        run_agent(input_file, model_filepath,output_file)
        print("RL Agent Done !")
    else:
        print("Invalid command. Use 'train' or 'test'.")