import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance
from embeddings_AETR_CLS import TreeAutoencoder,get_expression_cls_embedding,flatten_expr
from expression_manager import HierarchicalExpression
# Ensure the custom module "pytrs" is in your PYTHONPATH.
sys.path.append(os.path.abspath("./pytrs"))
from pytrs import (
    parse_sexpr,
    expr_to_str,
    calculate_cost,
    create_rules,
    generate_random_assignments,
    evaluate_expr
)

# Set number of threads and create the rules dictionary.
torch.set_num_threads(4)
rules = create_rules(vector_size=9)
if torch.cuda.is_available() and sys.argv[1].lower() == "train":
    device = torch.device("cuda")
else :
    device = torch.device("cpu")
embeddings_model = TreeAutoencoder()
embeddings_model.load_state_dict(torch.load("./model_Transformer_CLS_depth3_100-v2.pth", map_location=device))
embeddings_model.to(device)
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

        # For demonstration, let's do a small linear transform
        self.net = nn.Sequential(
            nn.Linear(self._embed_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, obs_dict: dict) -> torch.Tensor:
        x = obs_dict["observation"]
       
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

        # Feature extractor.
        self.features_extractor = CustomFeaturesExtractor(observation_space, features_dim)

        # Rule selection network.
        self.rule_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.rule_dim)
        )

        # Position selection network.
        self.position_net = nn.Sequential(
            nn.Linear(features_dim + self.rule_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.max_positions)
        )

        # Value network.
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Separate optimizers.
        self.rule_optimizer = optim.Adam(self.rule_net.parameters(), lr=lr)
        self.position_optimizer = optim.Adam(self.position_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-4)
        self.optimizer = self.rule_optimizer
        
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        device = next(self.parameters()).device
        # If observation is a dict, ensure each entry has a batch dimension.
        if isinstance(observation, dict):
            new_obs = {}
            for key, value in observation.items():
                # If it's a NumPy array and 1D, add a batch dimension.
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        value = np.expand_dims(value, axis=0)
                    new_obs[key] = torch.as_tensor(value, device=device)
                # If it's a torch tensor and 1D, unsqueeze it.
                elif torch.is_tensor(value):
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                    new_obs[key] = value.to(device)
                else:
                    new_obs[key] = value
            observation = new_obs
        else:
            if isinstance(observation, np.ndarray) and observation.ndim == 1:
                observation = np.expand_dims(observation, axis=0)
            observation = torch.as_tensor(observation, device=device)
        
        # Now call forward with the properly batched observation.
        action, _, _ = self.forward(observation, deterministic=deterministic)
        return action.cpu().numpy(), state
    
        

    
    def forward(self, obs, action_masks=None, deterministic=False):
        """
        Expects:
          - obs: a dict with keys "observation" and "action_mask".
          - action_masks: a tensor of shape [batch, total_actions].
        Returns:
          final_action, value, log_prob
        """
        if action_masks is None:
            if isinstance(obs, dict) and "action_mask" in obs:
                action_masks = obs["action_mask"]
            else:
                raise ValueError("Action masks must be provided!")
                
        features = self.features_extractor(obs)
        batch_size = features.shape[0]
        # Reshape the flat mask to [batch, rule_dim, max_positions]
        rule_masks = action_masks.view(batch_size, self.rule_dim, self.max_positions)

        # --- Rule Selection ---
        rule_logits = self.rule_net(features)
        valid_rules = rule_masks.bool().any(dim=2)
        masked_rule_logits = torch.where(valid_rules, rule_logits,
                                         torch.tensor(-float("inf"), device=rule_logits.device))
        rule_dist = Categorical(logits=masked_rule_logits)
        rule_action = rule_dist.mode if deterministic else rule_dist.sample()

        # --- Position Selection ---
        rule_oh = nn.functional.one_hot(rule_action, num_classes=self.rule_dim).float()
        position_input = torch.cat([features, rule_oh], dim=1)
        position_logits = self.position_net(position_input)
        position_mask = rule_masks[torch.arange(batch_size), rule_action]
        masked_position_logits = torch.where(position_mask.bool(), position_logits,
                                             torch.tensor(-float("inf"), device=position_logits.device))
        position_dist = Categorical(logits=masked_position_logits)
        position_action = position_dist.mode if deterministic else position_dist.sample()

        final_action = rule_action * self.max_positions + position_action
        values = self.value_net(features)
        log_prob = rule_dist.log_prob(rule_action) + position_dist.log_prob(position_action)
        return final_action, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        """
        Given observations and actions, compute value estimates, log probabilities, and entropy.
        """
        if action_masks is None:
            if isinstance(obs, dict) and "action_mask" in obs:
                action_masks = obs["action_mask"]
            else:
                raise ValueError("Action masks must be provided!")
        features = self.features_extractor(obs)
        batch_size = features.shape[0]
        rule_masks = action_masks.view(batch_size, self.rule_dim, self.max_positions)
        rule_logits = self.rule_net(features)
        valid_rules = rule_masks.bool().any(dim=2)
        masked_rule_logits = torch.where(valid_rules, rule_logits,
                                         torch.tensor(-float("inf"), device=rule_logits.device))
        rule_dist = Categorical(logits=masked_rule_logits)
        # Ensure indices are 1D Long tensors.
        rule_actions = (actions // self.max_positions).long().view(-1)
        position_actions = (actions % self.max_positions).long().view(-1)
        rule_oh = nn.functional.one_hot(rule_actions, num_classes=self.rule_dim).float()
        position_input = torch.cat([features, rule_oh], dim=1)
        position_logits = self.position_net(position_input)
        position_mask = rule_masks[torch.arange(batch_size), rule_actions]
        masked_position_logits = torch.where(position_mask.bool(), position_logits,
                                             torch.tensor(-float("inf"), device=position_logits.device))
        position_dist = Categorical(logits=masked_position_logits)
        log_prob = rule_dist.log_prob(rule_actions) + position_dist.log_prob(position_actions)
        entropy = self.ent_coef * (rule_dist.entropy().mean() + position_dist.entropy().mean())
        values = self.value_net(features)
        return values, log_prob, entropy

    def predict_values(self, obs, action_masks=None):
        """
        Returns the value estimates for the given observations.
        """
        if action_masks is None:
            if isinstance(obs, dict) and "action_mask" in obs:
                action_masks = obs["action_mask"]
            else:
                raise ValueError("Action masks must be provided!")
        features = self.features_extractor(obs)
        return self.value_net(features)

    def set_training_mode(self, mode: bool):
        self.train(mode)

############################################
# Custom PPO Implementation (HierarchicalPPO)#
############################################
class HierarchicalPPO(PPO):
    def __init__(self, *args, rule_dim=5, max_positions=2, **kwargs):
        """
        rule_dim: Total number of high-level rules.
        max_positions: Number of sub-actions (positions) available per rule.
        """
        if "policy_kwargs" not in kwargs:
            kwargs["policy_kwargs"] = {}
        kwargs["policy_kwargs"]["rule_dim"] = rule_dim
        kwargs["policy_kwargs"]["max_positions"] = max_positions
        super().__init__(*args, **kwargs)
        self.ep_info_buffer = []

    


    def _update_info_buffer(self, infos):
        """
        Updates the info buffer with relevant episodic information.
        """
        if hasattr(self, "ep_info_buffer"):
            for info in infos:
                maybe_ep_info = info.get("episode")
                if maybe_ep_info is not None:
                    self.ep_info_buffer.append(maybe_ep_info)

    def obs_to_tensor(self,obs, device):
        """
        Convert observation to PyTorch tensor.
        """
        if isinstance(obs, dict):
            return {
                key: torch.as_tensor(value).to(device)
                for key, value in obs.items()
            }
        return torch.as_tensor(obs).to(device)
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        print("called")
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        # Initialize episode info buffer for this collection phase
        self.ep_info_buffer = []
        print("Cleared ep_info_buffer")  # Debug print

        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = self.obs_to_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            step_result = env.step(clipped_actions)
            if len(step_result) == 4:
                new_obs, rewards, dones, infos = step_result
                truncated = dones
            else:
                new_obs, rewards, dones, truncated, infos = step_result

            self.num_timesteps += env.num_envs

            # Store episode info
            for info in infos:
                if "episode" in info:
                    #print(f"Found episode info: {info['episode']}")  # Debug print
                    self.ep_info_buffer.append(info["episode"])
                    #print(f"Current buffer size: {len(self.ep_info_buffer)}")  # Debug print


            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.obs_to_tensor(infos[idx]["terminal_observation"], self.device)
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            obs_tensor = self.obs_to_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        print(f"End of collection. Buffer size: {len(self.ep_info_buffer)}")  # Debug print
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        # Get rollout data
        self.policy.ent_coef = max(0.01, self.policy.ent_coef * 0.99)
        print(f"Starting training. Current buffer size: {len(self.ep_info_buffer)}")  # Debug print
        rollout_data_gen = self.rollout_buffer.get(batch_size=None)
        rollout_data = next(rollout_data_gen)
        
        if isinstance(rollout_data.observations, dict):
            observations = rollout_data.observations["observation"]
            action_masks = rollout_data.observations["action_mask"]
        else:
            observations = rollout_data.observations
            action_masks = None

        actions = rollout_data.actions
        old_values = rollout_data.old_values
        old_log_probs = rollout_data.old_log_prob
        advantages = rollout_data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rollout_data.returns

        values, new_log_probs, entropy = self.policy.evaluate_actions(
            {"observation": observations, "action_mask": action_masks},
            actions,
            action_masks=action_masks
        )

        # Calculate losses
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        ).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy_loss = -entropy.mean()
        total_loss = policy_loss + value_loss + entropy_loss

        # Optimization step
        self.policy.rule_optimizer.zero_grad()
        self.policy.position_optimizer.zero_grad()
        self.policy.value_optimizer.zero_grad()
        total_loss.backward()
        # Log episode info if available
        """if len(self.ep_info_buffer) > 0:
            ep_rewards = [ep_info["r"] for ep_info in self.ep_info_buffer]
            ep_lengths = [ep_info["l"] for ep_info in self.ep_info_buffer]
            mean_reward = np.mean(ep_rewards)
            mean_length = np.mean(ep_lengths)
            
            self.logger.record("rollout/ep_reward_mean", mean_reward)
            self.logger.record("rollout/ep_length_mean", mean_length)
        """
        # Rest of the training metrics
        grad_norm = 0.0
        for param in self.policy.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.logger.record("train/grad_norm", grad_norm)
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/approx_kl", (old_log_probs - new_log_probs).mean().item())
        self.logger.record("train/entropy_coef", self.policy.ent_coef)
        explained_var = explained_variance(old_values.detach().cpu().numpy().flatten(),
                                        returns.detach().cpu().numpy().flatten())
        self.logger.record("train/explained_variance", explained_var)

        # Perform optimization step
        self.policy.rule_optimizer.step()
        self.policy.position_optimizer.step()
        self.policy.value_optimizer.step()

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
        self.max_steps = 40
        self.max_expression_size = 10000
        self.initial_cost = 0
        self.embedding_dim = 256
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
        parsed = parse_sexpr(self.abstracted_expression)
        has_valid = False
        for rule_idx, rule_name in enumerate(self.rules):
            if rule_name == "END":
                mask[rule_idx * self.max_positions] = 1.0
                continue
            rule_obj = rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            valid_positions = min(len(matches), self.max_positions)
            if valid_positions > 0:
                has_valid = True
                start = rule_idx * self.max_positions
                mask[start:start + valid_positions] = 1.0
        if not has_valid and "END" in self.rules:
            end_rule_idx = self.rules.index("END")
            mask[end_rule_idx * self.max_positions] = 1.0
        return mask

    def _encode_expression(self, expr: str) -> np.ndarray:
        encoded = np.zeros(self.max_expression_size, dtype=np.uint8)
        bytes_data = expr.encode("ascii", "ignore")[:self.max_expression_size]
        encoded[:len(bytes_data)] = np.frombuffer(bytes_data, dtype=np.uint8)
        return encoded

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the pointer if it doesn't exist
        if not hasattr(self, "current_index"):
            self.current_index = 0

        # Use the next expression in order
        self.expression = self.expressions[self.current_index]
        # Increment and wrap the pointer
        self.current_index = (self.current_index + 1) % len(self.expressions)

        self.initial_expression = self.expression
        self.he = HierarchicalExpression(self.expression, min_depth=3)
        self.abstracted_expression = self.he.get_abstracted()
        self.sub_vecs = self.he.collect_all_vector_subexpr_strings()
        self.current_abstracted_index = 0
        self.steps = 0
        self.initial_cost = self.current_cost = self.get_cost(self.expression)
        return {
            "observation": self._embed_expression(self.abstracted_expression),
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
    
        print("Old expression : ",self.expression)
        print("Old abstracted expression : ",self.abstracted_expression)
        print("Old Cost : ",self.current_cost)
        if rule_name == "END":
            terminated = True
            reward = ( (self.initial_cost - self.current_cost) / self.initial_cost * 100  )
            #reward = 10 if self._valid_end_action() else -10
            isValid = self._valid_end_action(self.abstracted_expression)
            if(isValid):
                reward = reward +  10
            else:
                reward = reward - 10
            """if(len(self.sub_vecs)<=(self.current_abstracted_index+1)):
                
            else:
                isValid = self._valid_end_action(self.abstracted_expression)
                if(isValid):
                    reward = 1
                else:
                    reward = -1
                self.current_abstracted_index +=1;
                print("switching")
                self.abstracted_expression = self.sub_vecs[self.current_abstracted_index]
                placeholders = self.he.get_expandable_placeholders_in_subvec(self.abstracted_expression)
                for ph in placeholders:
                    self.he.expand_one_level(ph, new_min_depth=2)
                self.sub_vecs = self.he.collect_all_vector_subexpr_strings()
                self.abstracted_expression = self.sub_vecs[self.current_abstracted_index]"""
        else:
            expression_abstracted = self.he.get_abstracted()
            parsed = parse_sexpr(expression_abstracted)
            rule_obj = rules[rule_name]

            matches = rule_obj.find_matching_subexpressions(parsed)
            if pos_idx >= len(matches):
                reward = -10
            else:
                k, _ = matches[pos_idx]
                new_expr_tree = rule_obj.apply_rule(parsed, path=k)
                temp = expr_to_str(new_expr_tree)
                self.expression = self.he.expand_expr_with_placeholders(temp)
                self.he = HierarchicalExpression(self.expression, min_depth=3)
                self.abstracted_expression = self.he.get_abstracted()
                self.sub_vecs = self.he.collect_all_vector_subexpr_strings()
                new_cost = self.get_cost(self.expression)
                if self.current_cost == 0:
                    self.current_cost = 1 
                if self.initial_cost ==0:
                    self.initial_cost = 1
                reward =( ( self.current_cost - new_cost) / self.current_cost )
                self.current_cost = new_cost
                if(len(self.abstracted_expression)>1495):
                    terminated = True
                    reward = ( (self.initial_cost - self.current_cost) / self.initial_cost * 100  )
                if (self.steps >= self.max_steps):
                    reward = ( (self.initial_cost - self.current_cost) / self.initial_cost * 100  )
        info = {"expression": self.expression}
        print("New expression : ",self.expression)
        print("New abstracted expression : ",self.abstracted_expression)
        print("New Cost : ",self.current_cost)
        print("Reward : ",reward)
        print("Rule Name : ",rule_name)
        print("At Position : ",pos_idx)
        if reward < -100:
            reward = -100
        terminated = terminated or (self.steps >= self.max_steps)
        if terminated or truncated:
            info["episode"] = {
                "r": reward,
                "l": self.steps,
                "t": None
            }   

        return {
            "observation": self._embed_expression(self.abstracted_expression),
            "action_mask": self.get_action_mask()
        }, reward, terminated, truncated, {"expression": self.expression}

    def _valid_end_action(self,expr: str) -> bool:
        expr_tree = parse_sexpr(expr)
        action_mask = self.get_action_mask()
        isValid = True
        for i, rule_name in enumerate(self.rules):
            if rule_name == "END":
                continue
            rule_obj = rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(expr_tree)
            if len(matches) > 0:
                for match in matches:
                    k, _ = match
                    new_expr_tree = rule_obj.apply_rule(expr_tree, path=k)
                    if calculate_cost(new_expr_tree) < self.current_cost:
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
    with open(file_path, "r") as f:
        for line in f:
            exp_str = line.split("|||")[0].strip()
            if not exp_str:
                continue
            try:
                expr = parse_sexpr(exp_str)
                if not expr.validate_expression():
                    print("Invalid:", exp_str)
                    continue

                token_seq = get_token_sequence(exp_str)
                # Skip expression if its token sequence is in the validation set.
                if token_seq in validation_token_set:
                    continue

                # Only add if token sequence hasn't been seen yet.
                if token_seq not in unique_expressions:
                    unique_expressions[token_seq] = exp_str
            except Exception as e:
                print("Error parsing expression:", exp_str, e)
    
    print("Number of unique valid expressions (excluding validation):", len(unique_expressions))
    return list(unique_expressions.values())
    
from agent_logger import log_training_details


def linear_schedule(initial: float, final: float):
    return lambda p: final + (initial - final) * p

def train_agent(expressions_file: str, total_timesteps: int = 1000000):
    validation_exprs = load_expressions("test_expressions.txt")
    expressions = load_expressions(expressions_file,validation_exprs)
    rules_list = list(rules.keys()) + ["END"]
    env = fheEnv(rules_list, expressions, max_positions=2)
    lr_schedule = linear_schedule(2e-4, 1e-4)
    
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    run_name = f"model_{job_id}"
    
    tensorboard_log_dir = f"./tensorboard/{run_name}"
    
    model_params = {
        "policy": HierarchicalMaskablePolicy,
        "env": env,
        "learning_rate": lr_schedule,
         "n_steps":2048,
        "batch_size":128,
        "gamma":0.99,
        "gae_lambda":0.95,
        #"clip_range_vf":0.2,
        "n_epochs":10,
        "verbose":1,
        "ent_coef":0.1,
        "verbose": 1,
        "tensorboard_log": tensorboard_log_dir,
        "policy_kwargs": {
            "rule_dim": len(rules_list),
            "max_positions": 2,
            "ent_coef": 0.1,
        },
    }
    
    log_training_details(model_params, job_id, num_data=len(expressions), 
                         num_actions=len(rules_list), total_timesteps=total_timesteps, 
                         output_model_name=run_name, notes="Hierarchical PPO training with reward clipping at -100")
    
    model = HierarchicalPPO(
        policy=HierarchicalMaskablePolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        #clip_range_vf=0.2,
        verbose=1,
        ent_coef=0.1,
        #vf_coef=0.5,
        #max_grad_norm=0.5,
        tensorboard_log=tensorboard_log_dir,
        rule_dim=len(rules_list),
        max_positions=2,
        policy_kwargs={
            "rule_dim": len(rules_list),
            "max_positions": 2,
            "ent_coef": 0.1,
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

    env = fheEnv(rules_list, expressions, max_positions=2)
    
    model = HierarchicalPPO(
        policy=HierarchicalMaskablePolicy,
        env=env,
        rule_dim=len(rules_list),
        max_positions=2,
        policy_kwargs={
            "rule_dim": len(rules_list),
            "max_positions": 2,
            "ent_coef": 0.1,
        },
    )
    model = model.load(model_filepath)
    
    num_expr = len(expressions)
    for i in range(num_expr):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, (list, np.ndarray)):
                action = int(action.item())
            obs, reward, done, truncated, info = env.step(action)
        results.append({
            "Test Expression": env.initial_expression,
            "Final Expression": info["expression"],
            "Initial Cost": env.initial_cost,
            "Final Cost": env.current_cost,
            "Steps": env.steps
        })
    
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    sheet_name = f"HierarchicalPPO_Test_{job_id}"
    log_test_results(results, sheet_name=sheet_name)
    
    return results
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test] [model_filepath (for test)]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    if mode == "train":
        train_agent("expressions.txt", total_timesteps=2000000)
    elif mode == "test":
        if len(sys.argv) < 3:
            print("Usage: python main.py test [model_filepath]")
            sys.exit(1)
        model_filepath = sys.argv[2]
        test_agent("test_expressions.txt", model_filepath)
    else:
        print("Invalid command. Use 'train' or 'test'.")