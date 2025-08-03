#!/usr/bin/env python3
"""
Actor-Critic models for 2048 game integration with Expectimax algorithm.

This module provides:
1. Critic network for board state evaluation (replaces hand-crafted heuristics)
2. Actor network for action probability prediction (guides move selection)
3. Training infrastructure for both networks
4. Integration utilities for Expectimax algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from game import Game2048, Direction, GameState
from experience_replay import PrioritizedReplayBuffer, Experience


class BoardEncoder:
    """Encodes 2048 board states into neural network input format."""
    
    @staticmethod
    def encode_board(board: List[List[int]]) -> torch.Tensor:
        """
        Encode a 2048 board into a tensor representation.
        
        Args:
            board: 4x4 board with tile values
            
        Returns:
            torch.Tensor: Encoded board representation (shape: [1, channels, 4, 4])
        """
        # Use log2 representation with multiple channels for different tile values
        encoded = np.zeros((16, 4, 4), dtype=np.float32)
        
        for i in range(4):
            for j in range(4):
                if board[i][j] > 0:
                    # Use log2 of the tile value as the channel index
                    tile_log = int(np.log2(board[i][j]))
                    if tile_log < 16:  # Max tile is 2^15 = 32768
                        encoded[tile_log, i, j] = 1.0
        
        return torch.FloatTensor(encoded).unsqueeze(0)  # Add batch dimension
    
    @staticmethod
    def encode_board_flat(board: List[List[int]]) -> torch.Tensor:
        """
        Encode board as a flat vector for simpler networks.
        
        Args:
            board: 4x4 board with tile values
            
        Returns:
            torch.Tensor: Flattened board representation (shape: [16])
        """
        flat_board = []
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    flat_board.append(0.0)
                else:
                    # Use log2 normalization
                    flat_board.append(np.log2(board[i][j]) / 16.0)
        
        return torch.FloatTensor(flat_board)

class CriticNetwork(nn.Module):
    """
    Critic network that evaluates board states.
    Replaces hand-crafted heuristics in Expectimax leaf evaluation.
    """
    
    def __init__(self, input_size: int = 16, hidden_size: int = 512):
        """
        Initialize the Critic network.
        
        Args:
            input_size: Size of flattened board representation
            hidden_size: Size of hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Single value output
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Args:
            state: Encoded board state
            
        Returns:
            torch.Tensor: Estimated state value
        """
        return self.network(state)
    
    def evaluate_board(self, board: List[List[int]]) -> float:
        """
        Evaluate a 2048 board state.
        
        Args:
            board: 4x4 board with tile values
            
        Returns:
            float: Estimated state value
        """
        with torch.no_grad():
            state = BoardEncoder.encode_board_flat(board)
            value = self.forward(state)
            return value.item()

class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities.
    Used to guide move selection and pruning in Expectimax.
    """
    
    def __init__(self, input_size: int = 16, hidden_size: int = 256, num_actions: int = 4):
        """
        Initialize the Actor network.
        
        Args:
            input_size: Size of flattened board representation
            hidden_size: Size of hidden layers
            num_actions: Number of possible actions (4 directions)
        """
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.
        
        Args:
            state: Encoded board state
            
        Returns:
            torch.Tensor: Action logits
        """
        return self.network(state)
    
    def get_action_probabilities(self, board: List[List[int]], 
                               valid_moves: List[Direction]) -> Dict[Direction, float]:
        """
        Get action probabilities for valid moves.
        
        Args:
            board: 4x4 board with tile values
            valid_moves: List of valid directions
            
        Returns:
            Dict[Direction, float]: Mapping of directions to probabilities
        """
        with torch.no_grad():
            state = BoardEncoder.encode_board_flat(board)
            logits = self.forward(state)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=0)
            
            # Map to direction indices
            direction_to_idx = {
                Direction.UP: 0,
                Direction.DOWN: 1,
                Direction.LEFT: 2,
                Direction.RIGHT: 3
            }
            
            # Filter and normalize for valid moves only
            valid_probs = {}
            total_prob = 0.0
            
            for direction in valid_moves:
                idx = direction_to_idx[direction]
                valid_probs[direction] = probs[idx].item()
                total_prob += valid_probs[direction]
            
            # Normalize
            if total_prob > 0:
                for direction in valid_probs:
                    valid_probs[direction] /= total_prob
            else:
                # Uniform distribution if all probabilities are zero
                uniform_prob = 1.0 / len(valid_moves)
                for direction in valid_moves:
                    valid_probs[direction] = uniform_prob
            
            return valid_probs

class ActorCriticTrainer:
    """
    Trainer for Actor-Critic networks using game experience.
    """
    
    def __init__(self, actor: ActorNetwork, critic: CriticNetwork,
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99):
        """
        Initialize the trainer.
        
        Args:
            actor: Actor network to train
            critic: Critic network to train
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
        """
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        
        # Optimizers with scheduling
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.9995)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9995)
        
        # Prioritized Experience Replay buffer with better settings
        self.experience_buffer = PrioritizedReplayBuffer(
            capacity=100_000,
            alpha=0.5,  # Less aggressive prioritization 
            beta=0.4,   # Initial importance sampling correction
            beta_increment_per_sampling=5e-5  # Slower beta growth
        )
        
        # Entropy scheduling for exploration
        self.initial_entropy_coeff = 0.1  # Higher initial value
        self.min_entropy_coeff = 0.001    # Minimum to maintain some exploration
        self.entropy_decay = 0.9999       # Very slow decay
        self.current_entropy_coeff = self.initial_entropy_coeff
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.training_steps = 0
    
    def collect_experience(self, experience: Experience):
        """Collect an experience for training."""
        self.experience_buffer.push(experience)

    def clear_experience(self):
        """Clear the experience buffer."""
        self.experience_buffer.clear()
    
    def get_training_info(self) -> Dict:
        """Get current training diagnostics."""
        return {
            'training_steps': self.training_steps,
            'current_entropy_coeff': self.current_entropy_coeff,
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
            'buffer_size': len(self.experience_buffer)
        }
    
    def train_step(self, batch_size: int = 32) -> Tuple[float, float, float]:
        """
        Perform one training step using prioritized experience replay.
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Tuple[float, float, float]: Actor loss, Critic loss, Actor entropy
        """
        if len(self.experience_buffer) < batch_size:
            return 0.0, 0.0, 0.0
        
        # Sample batch from prioritized buffer
        batch, indices, is_weights = self.experience_buffer.sample(batch_size)
        is_weights = torch.FloatTensor(is_weights)
        
        # Prepare batch data
        states = torch.stack([BoardEncoder.encode_board_flat(exp.state) for exp in batch])
        actions = torch.LongTensor([self._direction_to_idx(exp.action) for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.stack([BoardEncoder.encode_board_flat(exp.next_state) for exp in batch])
        dones = torch.BoolTensor([exp.done for exp in batch])
        
        # Train Critic and get TD errors for priority updates
        critic_loss, td_errors = self._train_critic(states, rewards, next_states, dones, is_weights)
        
        # Update priorities based on TD errors
        self.experience_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Train Actor
        actor_loss, entropy = self._train_actor(states, actions, rewards, next_states, dones, is_weights)
        
        # Update training step counter
        self.training_steps += 1
        
        # Step critic scheduler occasionally
        if self.training_steps % 100 == 0:
            self.critic_scheduler.step()
        
        return actor_loss, critic_loss, entropy
    
    def compute_shaped_reward(
        self,
        prev_board,
        new_board,
        prev_score,
        new_score,
        prev_max_tile,
        new_max_tile,
        success,
        game_done
    ):
        # Core score difference
        reward = (new_score - prev_score) / 100.0  # scale down

        # Count empty tiles
        prev_empty = sum(row.count(0) for row in prev_board)
        new_empty = sum(row.count(0) for row in new_board)
        empty_diff = new_empty - prev_empty
        reward += 0.1 * empty_diff  # encourages board space

        # Max tile increase
        if new_max_tile > prev_max_tile:
            reward += (new_max_tile - prev_max_tile) / 10.0  # milestone bonus

        # Penalize failed move
        if not success:
            reward -= 1.0

        # Game over penalty
        if game_done:
            reward -= 10.0

        return reward
    
    def _train_critic(self, states: torch.Tensor, rewards: torch.Tensor,
                     next_states: torch.Tensor, dones: torch.Tensor, 
                     is_weights: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Train the critic network using TD error with importance sampling."""
        # Current state values
        current_values = self.critic(states).squeeze()
        
        # Target values
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            targets = rewards + self.gamma * next_values * (~dones)
            
            # Clip targets to reasonable range to prevent exploding values
            targets = torch.clamp(targets, -100, 100)
        
        # Compute TD errors
        td_errors = targets - current_values
        
        # Compute weighted loss using importance sampling weights
        critic_loss = (is_weights * td_errors.pow(2)).mean()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # Add gradient clipping
        self.critic_optimizer.step()
        
        loss_value = critic_loss.item()
        self.critic_losses.append(loss_value)
        
        # Return loss and TD errors (for priority updates)
        return loss_value, td_errors.abs()
    
    def _train_actor(self, states: torch.Tensor, actions: torch.Tensor,
                    rewards: torch.Tensor, next_states: torch.Tensor, 
                    dones: torch.Tensor, is_weights: torch.Tensor) -> Tuple[float, float]:
        """Train the actor network using policy gradient with importance sampling."""
        # Get action logits and distribution
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Compute advantages (TD error) with normalization
        with torch.no_grad():
            current_values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            targets = rewards + self.gamma * next_values * (~dones)
            advantages = targets - current_values
            
            # Normalize advantages for stability
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()
        
        # Policy gradient loss with importance sampling weights
        log_probs = dist.log_prob(actions)
        pg_loss = -(is_weights * log_probs * advantages).mean()
        entropy = dist.entropy().mean()
        
        # Use adaptive entropy coefficient
        actor_loss = pg_loss - self.current_entropy_coeff * entropy
        
        # Update entropy coefficient (decay slowly)
        self.current_entropy_coeff = max(
            self.min_entropy_coeff,
            self.current_entropy_coeff * self.entropy_decay
        )
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # Reduced clip value
        self.actor_optimizer.step()
        
        # Step learning rate scheduler occasionally
        if self.training_steps % 100 == 0:
            self.actor_scheduler.step()
        
        loss_value = actor_loss.item()
        self.actor_losses.append(loss_value)
        entropy_value = entropy.item()
        return loss_value, entropy_value
    
    def _direction_to_idx(self, direction: Direction) -> int:
        """Convert Direction to index."""
        mapping = {
            Direction.UP: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 2,
            Direction.RIGHT: 3
        }
        return mapping[direction]


class ExpectimaxIntegrator:
    """
    Utilities for integrating Actor-Critic with Expectimax algorithm.
    """
    
    def __init__(self, actor: ActorNetwork, critic: CriticNetwork):
        """
        Initialize the integrator.
        
        Args:
            actor: Trained actor network
            critic: Trained critic network
        """
        self.actor = actor
        self.critic = critic
    
    def evaluate_leaf_state(self, game: Game2048) -> float:
        """
        Use critic network as leaf evaluator for Expectimax.
        
        Args:
            game: Game state to evaluate
            
        Returns:
            float: Estimated state value
        """
        return self.critic.evaluate_board(game.get_board())
    
    def get_ordered_moves(self, game: Game2048, top_k: Optional[int] = None) -> List[Direction]:
        """
        Get moves ordered by actor network probabilities.
        Can be used for move ordering or top-k pruning in Expectimax.
        
        Args:
            game: Current game state
            top_k: If specified, return only top-k moves
            
        Returns:
            List[Direction]: Moves ordered by decreasing probability
        """
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return []
        
        # Get action probabilities
        action_probs = self.actor.get_action_probabilities(game.get_board(), valid_moves)
        
        # Sort by probability (descending)
        sorted_moves = sorted(valid_moves, key=lambda move: action_probs[move], reverse=True)
        
        # Apply top-k filtering if specified
        if top_k is not None:
            sorted_moves = sorted_moves[:top_k]
        
        return sorted_moves
    
    def should_prune_move(self, game: Game2048, move: Direction, 
                         prob_threshold: float = 0.1) -> bool:
        """
        Determine if a move should be pruned based on low actor probability.
        
        Args:
            game: Current game state
            move: Move to check
            prob_threshold: Minimum probability threshold
            
        Returns:
            bool: True if move should be pruned
        """
        valid_moves = game.get_valid_moves()
        
        if move not in valid_moves:
            return True
        
        action_probs = self.actor.get_action_probabilities(game.get_board(), valid_moves)
        return action_probs[move] < prob_threshold


class GameDataCollector:
    """
    Collects training data from game simulations.
    """
    
    def __init__(self, trainer: ActorCriticTrainer):
        """
        Initialize the data collector.
        
        Args:
            trainer: Actor-Critic trainer to collect data for
        """
        self.trainer = trainer
    
    def collect_experience(self,
        current_state,
        new_state,
        current_score,
        new_score,
        current_max_tile,
        new_max_tile,
        success,
        game_done,
        move):
        
        # Always calculate reward, including for failed moves
        reward = self.trainer.compute_shaped_reward(
            current_state,
            new_state,
            current_score,
            new_score,
            current_max_tile,
            new_max_tile,
            success,
            game_done
        )
        
        # Create experience for both successful and failed moves
        experience = Experience(current_state, move, reward, new_state, game_done)
        
        # Add to trainer's buffer
        self.trainer.collect_experience(experience)
    
    def _copy_game(self, game: Game2048) -> Game2048:
        """Create a deep copy of the game state."""
        new_game = Game2048(size=game.size)
        new_game.board = [row[:] for row in game.board]
        new_game.score = game.score
        new_game.game_state = game.game_state
        new_game.move_count = game.move_count
        return new_game


def create_actor_critic_models(hidden_size: int = 512, 
                              actor_lr: float = 1e-4, 
                              critic_lr: float = 1e-4) -> Tuple[ActorNetwork, CriticNetwork, ActorCriticTrainer]:
    """
    Factory function to create Actor-Critic models and trainer with prioritized experience replay.
    
    Args:
        hidden_size: Size of hidden layers
        actor_lr: Learning rate for actor network
        critic_lr: Learning rate for critic network
        
    Returns:
        Tuple[ActorNetwork, CriticNetwork, ActorCriticTrainer]: Created models and trainer
    """
    actor = ActorNetwork(hidden_size=hidden_size // 2)
    critic = CriticNetwork(hidden_size=hidden_size)
    trainer = ActorCriticTrainer(actor, critic, actor_lr=actor_lr, critic_lr=critic_lr)
    
    return actor, critic, trainer


def save_models(actor: ActorNetwork, critic: CriticNetwork, filepath_prefix: str):
    """
    Save trained models to disk.
    
    Args:
        actor: Actor network to save
        critic: Critic network to save
        filepath_prefix: Prefix for save files
    """
    torch.save(actor.state_dict(), f"{filepath_prefix}_actor.pth")
    torch.save(critic.state_dict(), f"{filepath_prefix}_critic.pth")


def load_models(filepath_prefix: str, hidden_size: int = 512) -> Tuple[ActorNetwork, CriticNetwork]:
    """
    Load trained models from disk.
    
    Args:
        filepath_prefix: Prefix for save files
        hidden_size: Size of hidden layers
        
    Returns:
        Tuple[ActorNetwork, CriticNetwork]: Loaded models
    """
    actor = ActorNetwork(hidden_size=hidden_size // 2)
    critic = CriticNetwork(hidden_size=hidden_size)
    
    actor.load_state_dict(torch.load(f"{filepath_prefix}_actor.pth"))
    critic.load_state_dict(torch.load(f"{filepath_prefix}_critic.pth"))
    
    actor.eval()
    critic.eval()
    
    return actor, critic


# Example usage and testing
if __name__ == "__main__":
    # Create models
    actor, critic, trainer = create_actor_critic_models()
    
    # Create a sample game
    game = Game2048()
    
    # Test critic evaluation
    print("Testing Critic Network:")
    board_value = critic.evaluate_board(game.get_board())
    print(f"Initial board value: {board_value:.4f}")
    
    # Test actor probabilities
    print("\nTesting Actor Network:")
    valid_moves = game.get_valid_moves()
    action_probs = actor.get_action_probabilities(game.get_board(), valid_moves)
    print("Action probabilities:")
    for direction, prob in action_probs.items():
        print(f"  {direction.value}: {prob:.4f}")
    
    # Test integration utilities
    print("\nTesting Integration Utilities:")
    integrator = ExpectimaxIntegrator(actor, critic)
    ordered_moves = integrator.get_ordered_moves(game, top_k=2)
    print(f"Top 2 moves: {[move.value for move in ordered_moves]}")
    
    print("\nActor-Critic models ready for integration!") 