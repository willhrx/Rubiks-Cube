"""PPO trainer for Rubik's Cube."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from tqdm import tqdm

from .trainer import CubeTrainer
from ..model.policy import CubePolicy
from ..model.environment import RubiksCubeEnv


class PPOTrainer(CubeTrainer):
    """
    Proximal Policy Optimization trainer for Rubik's Cube.
    
    This class implements the PPO algorithm for training Rubik's Cube solvers.
    """
    
    def __init__(self, 
                 policy: CubePolicy,
                 env: RubiksCubeEnv,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 3e-4,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 num_epochs: int = 4,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the PPO trainer.
        
        Args:
            policy: Policy network to train
            env: Environment to train in
            device: Device to use for training
            learning_rate: Learning rate for the optimizer
            clip_ratio: PPO clip ratio
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs to train on each batch of data
            batch_size: Batch size for training
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            log_dir: Directory to save TensorBoard logs
            checkpoint_dir: Directory to save model checkpoints
        """
        super(PPOTrainer, self).__init__(
            policy=policy,
            env=env,
            device=device,
            learning_rate=learning_rate,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir
        )
        
        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def collect_trajectory(self, max_steps: int) -> Dict[str, Any]:
        """
        Collect a trajectory from the environment.
        
        Args:
            max_steps: Maximum number of steps to collect
            
        Returns:
            Dictionary of trajectory data
        """
        # Initialize lists to store trajectory data
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Reset the environment
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        num_episodes = 0
        
        # Collect trajectory data
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action and value from policy
            with torch.no_grad():
                action_probs, value = self.policy(state_tensor)
                
                # Sample action from the distribution
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample().item()
                log_prob = action_distribution.log_prob(torch.tensor([action], device=self.device)).item()
            
            # Take a step in the environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_done = terminated or truncated
            
            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)
            dones.append(done)
            
            # Update state and done flag
            state = next_state
            done = next_done
            
            # Update episode stats
            episode_reward += reward
            episode_length += 1
            
            # If the episode is done, reset the environment
            if done:
                num_episodes += 1
                self.episodes += 1
                
                # Log episode data
                self.writer.add_scalar('Train/EpisodeReward', episode_reward, self.episodes)
                self.writer.add_scalar('Train/EpisodeLength', episode_length, self.episodes)
                self.writer.add_scalar('Train/SolvedFlag', 1 if info['solved'] else 0, self.episodes)
                
                if info['solved']:
                    self.writer.add_scalar('Train/StepsToSolve', episode_length, self.episodes)
                
                # Reset episode stats
                state, _ = self.env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(
            rewards, values, dones
        )
        
        # Convert lists to tensors
        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)
        advantages = advantages.astype(np.float32)
        returns = returns.astype(np.float32)
        
        # Update step count
        self.steps += max_steps
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
            'num_episodes': num_episodes
        }
    
    def _compute_advantages_and_returns(self, rewards: List[float], 
                                      values: List[float], 
                                      dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        # Convert to numpy arrays
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        
        # Get trajectory length
        traj_length = len(rewards)
        
        # Initialize arrays
        advantages = np.zeros(traj_length, dtype=np.float32)
        returns = np.zeros(traj_length, dtype=np.float32)
        
        # Set the last values
        last_value = 0  # Zero value for terminal state
        last_advantage = 0
        
        # Compute GAE
        for t in reversed(range(traj_length)):
            # If this is the last step in an episode, bootstrap with 0
            if t == traj_length - 1 or dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # Calculate TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # Calculate advantage
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            
            # Store for next iteration
            last_advantage = advantages[t]
            
            # Calculate returns
            returns[t] = rewards[t] + self.gamma * (1 - dones[t]) * (returns[t + 1] if t < traj_length - 1 else 0)
        
        return advantages, returns
    
    def train(self, num_iterations: int, steps_per_iteration: int, eval_interval: int = 10) -> Dict[str, Any]:
        """
        Train the policy using PPO.
        
        Args:
            num_iterations: Number of iterations to train for
            steps_per_iteration: Number of steps to collect per iteration
            eval_interval: Interval (in iterations) at which to evaluate the policy
            
        Returns:
            Dictionary of training statistics
        """
        start_time = time.time()
        best_solve_rate = 0.0
        
        for iteration in range(num_iterations):
            # Collect trajectory
            trajectory = self.collect_trajectory(steps_per_iteration)
            
            # Optimize policy
            train_metrics = self._optimize_policy(trajectory)
            
            # Log training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, iteration)
            
            # Evaluate the policy
            if (iteration + 1) % eval_interval == 0:
                print(f"\nEvaluating policy after iteration {iteration+1}...")
                eval_metrics = self.evaluate()
                
                print(f"Eval metrics: {eval_metrics}")
                
                # Save checkpoint if best
                if eval_metrics['solve_rate'] > best_solve_rate:
                    best_solve_rate = eval_metrics['solve_rate']
                    self.save_checkpoint(is_best=True)
                    print(f"New best solve rate: {best_solve_rate:.2f}")
                
                # Regular checkpoint
                if (iteration + 1) % (eval_interval * 5) == 0:
                    self.save_checkpoint()
            
            # Print progress
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(trajectory['returns'])
            
            print(f"Iteration {iteration+1}/{num_iterations}, "
                  f"Avg Return: {avg_reward:.2f}, "
                  f"Episodes: {self.episodes}, "
                  f"Steps: {self.steps}, "
                  f"Time: {elapsed_time:.2f}s")
        
        # Save final checkpoint
        self.save_checkpoint()
        
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'best_solve_rate': best_solve_rate,
            'training_time': time.time() - start_time
        }
    
    def _optimize_policy(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize the policy using PPO.
        
        Args:
            trajectory: Dictionary of trajectory data
            
        Returns:
            Dictionary of training metrics
        """
        states = trajectory['states']
        actions = trajectory['actions']
        old_log_probs = trajectory['log_probs']
        advantages = trajectory['advantages']
        returns = trajectory['returns']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Calculate number of mini-batches
        batch_size = len(states)
        mini_batch_size = min(self.batch_size, batch_size)
        num_mini_batches = batch_size // mini_batch_size
        
        # Training metrics
        metrics = {
            'PolicyLoss': 0.0,
            'ValueLoss': 0.0,
            'EntropyLoss': 0.0,
            'TotalLoss': 0.0,
            'ClipFraction': 0.0,
            'ApproxKL': 0.0
        }
        
        # Train for multiple epochs
        for epoch in range(self.num_epochs):
            # Shuffle the data
            indices = np.random.permutation(batch_size)
            states_shuffled = states_tensor[indices]
            actions_shuffled = actions_tensor[indices]
            old_log_probs_shuffled = old_log_probs_tensor[indices]
            advantages_shuffled = advantages_tensor[indices]
            returns_shuffled = returns_tensor[indices]
            
            # Train on mini-batches
            for i in range(num_mini_batches):
                # Get mini-batch
                start_idx = i * mini_batch_size
                end_idx = min(start_idx + mini_batch_size, batch_size)
                
                states_batch = states_shuffled[start_idx:end_idx]
                actions_batch = actions_shuffled[start_idx:end_idx]
                old_log_probs_batch = old_log_probs_shuffled[start_idx:end_idx]
                advantages_batch = advantages_shuffled[start_idx:end_idx]
                returns_batch = returns_shuffled[start_idx:end_idx]
                
                # Forward pass
                action_probs, values = self.policy(states_batch)
                
                # Create action distribution
                dist = torch.distributions.Categorical(action_probs)
                
                # Calculate new log probs
                new_log_probs = dist.log_prob(actions_batch)
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                
                # Calculate surrogate objectives
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(-1), returns_batch)
                
                # Calculate entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Calculate total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Calculate metrics for logging
                with torch.no_grad():
                    # Approximate KL divergence
                    kl = (old_log_probs_batch - new_log_probs).mean().item()
                    
                    # Clipping fraction
                    clip_frac = ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                    
                    # Update metrics
                    metrics['PolicyLoss'] += policy_loss.item() / (self.num_epochs * num_mini_batches)
                    metrics['ValueLoss'] += value_loss.item() / (self.num_epochs * num_mini_batches)
                    metrics['EntropyLoss'] += entropy_loss.item() / (self.num_epochs * num_mini_batches)
                    metrics['TotalLoss'] += loss.item() / (self.num_epochs * num_mini_batches)
                    metrics['ClipFraction'] += clip_frac / (self.num_epochs * num_mini_batches)
                    metrics['ApproxKL'] += kl / (self.num_epochs * num_mini_batches)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return metrics