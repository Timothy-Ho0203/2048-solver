#!/usr/bin/env python3
"""
Demo script for the 2048 game implementation.
Shows both human gameplay and agent interaction examples.
"""

import random
import time
from typing import List, Dict, Optional
from game import Game2048, Direction, GameState
from display import Display2048, print_game

# Actor-Critic integration
from actor_critic import (
    create_actor_critic_models, ExpectimaxIntegrator, 
    ActorCriticTrainer, GameDataCollector, save_models, load_models
)
from experience_replay import Experience
from config import Config


class SimpleAgent:
    """
    An AI agent that uses the expectimax algorithm to play 2048.
    Expectimax is ideal for 2048 as it handles the probabilistic nature of tile generation.
    """
    
    def __init__(self, game: Game2048, depth: int = 3):
        """
        Initialize the agent.
        
        Args:
            game: Game2048 instance to play
            depth: Maximum search depth for expectimax algorithm
        """
        self.game = game
        self.name = "Expectimax Agent"
        self.depth = depth
    
    def get_move(self) -> Optional[Direction]:
        """
        Get the next move using expectimax algorithm.
        
        Returns:
            Optional[Direction]: Next move or None if no moves available
        """
        valid_moves = self.game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        best_move = valid_moves[0]  # Default to first valid move
        best_score = float('-inf')
        
        for move in valid_moves:
            # Create a copy of the game to test the move
            test_game = self._copy_game_state(self.game)
            if test_game.move(move):
                # Use expectimax to evaluate this move
                score = self._expectimax(test_game, self.depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    
    def _expectimax(self, game: Game2048, depth: int, is_max_node: bool) -> float:
        """
        Expectimax algorithm implementation.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            is_max_node: True for MAX nodes (player moves), False for CHANCE nodes (tile placement)
            
        Returns:
            float: Expected score for this node
        """
        # Terminal conditions
        if depth == 0 or game.get_game_state() != GameState.ONGOING:
            return self._evaluate_board(game)
        
        if is_max_node:
            # MAX node: Player's turn, maximize score
            max_score = float('-inf')
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                return self._evaluate_board(game)
            
            for move in valid_moves:
                test_game = self._copy_game_state(game)
                if test_game.move(move):
                    score = self._expectimax(test_game, depth - 1, False)
                    max_score = max(max_score, score)
            
            return max_score
        else:
            # CHANCE node: Random tile placement, calculate expected value
            empty_cells = [(i, j) for i in range(game.size) 
                          for j in range(game.size) if game.board[i][j] == 0]
            
            if not empty_cells:
                return self._evaluate_board(game)
            
            expected_score = 0.0
            
            # For each empty cell, consider placing a 2 (90%) or 4 (10%)
            for row, col in empty_cells:
                # Place tile with value 2 (90% probability)
                test_game_2 = self._copy_game_state(game)
                test_game_2.board[row][col] = 2
                score_2 = self._expectimax(test_game_2, depth - 1, True)
                
                # Place tile with value 4 (10% probability)
                test_game_4 = self._copy_game_state(game)
                test_game_4.board[row][col] = 4
                score_4 = self._expectimax(test_game_4, depth - 1, True)
                
                # Expected value for this cell
                cell_expected = 0.9 * score_2 + 0.1 * score_4
                expected_score += cell_expected
            
            # Average over all possible placements
            return expected_score / len(empty_cells)
    
    def _evaluate_board(self, game: Game2048) -> float:
        """
        Evaluate the current board state using multiple heuristics.
        
        Args:
            game: Game2048 instance to evaluate
            
        Returns:
            float: Board evaluation score
        """
        if game.get_game_state() == GameState.LOST:
            return float('-inf')
        
        board = game.get_board()
        score = 0.0
        
        # 1. Current game score (weighted)
        score += game.get_score() * 1.0
        
        # 2. Number of empty cells (more empty cells = better)
        empty_cells = sum(1 for i in range(game.size) 
                         for j in range(game.size) if board[i][j] == 0)
        score += empty_cells * 100.0
        
        # 3. Maximum tile value
        max_tile = max(max(row) for row in board)
        score += max_tile * 10.0
        
        # 4. Monotonicity (tiles should increase in one direction)
        score += self._monotonicity_score(board) * 50.0
        
        # 5. Smoothness (similar tiles should be adjacent)
        score += self._smoothness_score(board) * 10.0
        
        # 6. Corner heuristic (keep largest tile in corner)
        score += self._corner_score(board) * 25.0
        
        return score
    
    def _monotonicity_score(self, board: List[List[int]]) -> float:
        """Calculate monotonicity score - how well tiles are ordered."""
        totals = [0, 0, 0, 0]  # up, down, left, right
        
        for i in range(4):
            current = 0
            next_val = 1
            while next_val < 4:
                while next_val < 4 and board[i][next_val] == 0:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                
                current_val = board[i][current] if board[i][current] != 0 else 0
                next_val_val = board[i][next_val] if board[i][next_val] != 0 else 0
                
                if current_val > next_val_val:
                    totals[0] += next_val_val - current_val
                elif next_val_val > current_val:
                    totals[1] += current_val - next_val_val
                
                current = next_val
                next_val += 1
        
        for j in range(4):
            current = 0
            next_val = 1
            while next_val < 4:
                while next_val < 4 and board[next_val][j] == 0:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                
                current_val = board[current][j] if board[current][j] != 0 else 0
                next_val_val = board[next_val][j] if board[next_val][j] != 0 else 0
                
                if current_val > next_val_val:
                    totals[2] += next_val_val - current_val
                elif next_val_val > current_val:
                    totals[3] += current_val - next_val_val
                
                current = next_val
                next_val += 1
        
        return max(totals[0], totals[1]) + max(totals[2], totals[3])
    
    def _smoothness_score(self, board: List[List[int]]) -> float:
        """Calculate smoothness score - how similar adjacent tiles are."""
        smoothness = 0.0
        
        for i in range(4):
            for j in range(4):
                if board[i][j] != 0:
                    value = board[i][j]
                    # Check right neighbor
                    if j < 3 and board[i][j + 1] != 0:
                        smoothness -= abs(value - board[i][j + 1])
                    # Check down neighbor
                    if i < 3 and board[i + 1][j] != 0:
                        smoothness -= abs(value - board[i + 1][j])
        
        return smoothness
    
    def _corner_score(self, board: List[List[int]]) -> float:
        """Calculate corner score - reward keeping max tile in corner."""
        max_tile = max(max(row) for row in board)
        
        # Check if max tile is in a corner
        corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
        
        if max_tile in corners:
            return max_tile
        
        # Penalize if max tile is not in corner
        return -max_tile * 0.1
    
    def _copy_game_state(self, game: Game2048) -> Game2048:
        """Create a deep copy of the game state."""
        new_game = Game2048(size=game.size)
        new_game.board = [row[:] for row in game.board]
        new_game.score = game.score
        new_game.game_state = game.game_state
        new_game.move_count = game.move_count
        return new_game
    
    def play_game(self, max_moves: int = 1000, display_moves: bool = True) -> Dict:
        """
        Play a complete game using the agent's strategy.
        
        Args:
            max_moves: Maximum number of moves to play
            display_moves: Whether to display each move
            
        Returns:
            Dict: Game statistics
        """
        display = Display2048(use_colors=True)
        moves_made = 0
        
        if display_moves:
            print(f"Starting game with {self.name}")
            print("=" * 50)
            display.print_board(self.game)
            time.sleep(1)
        
        while (self.game.get_game_state() == GameState.ONGOING):
            
            move = self.get_move()
            
            if move is None:
                break
            
            success = self.game.move(move)
            
            if success:
                moves_made += 1
                
                if display_moves:
                    print(f"\nMove {moves_made}: {move.value}")
                    display.print_board(self.game)
                    time.sleep(0.5)
            else:
                break
        
        # Game finished
        final_info = self.game.get_game_info()
        
        if display_moves:
            print("\n" + "=" * 50)
            print("GAME FINISHED!")
            print(f"Final Score: {final_info['score']}")
            print(f"Max Tile: {final_info['max_tile']}")
            print(f"Total Moves: {final_info['move_count']}")
            print(f"Game State: {final_info['game_state'].value}")
        
        return {
            'agent_name': self.name,
            'final_score': final_info['score'],
            'max_tile': final_info['max_tile'],
            'move_count': final_info['move_count'],
            'game_state': final_info['game_state'],
            'moves_made': moves_made
        }


class RandomAgent:
    """
    A random agent that makes random valid moves.
    Useful for testing and comparison.
    """
    
    def __init__(self, game: Game2048):
        """Initialize the random agent."""
        self.game = game
        self.name = "Random Agent"
    
    def get_move(self) -> Optional[Direction]:
        """Get a random valid move."""
        valid_moves = self.game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        return random.choice(valid_moves)
    
    def play_game(self, max_moves: int = 1000, display_moves: bool = False) -> Dict:
        """Play a complete game with random moves."""
        moves_made = 0
        
        while (self.game.get_game_state() == GameState.ONGOING and 
               moves_made < max_moves):
            
            move = self.get_move()
            
            if move is None:
                break
            
            success = self.game.move(move)
            
            if success:
                moves_made += 1
            else:
                break
        
        final_info = self.game.get_game_info()
        
        return {
            'agent_name': self.name,
            'final_score': final_info['score'],
            'max_tile': final_info['max_tile'],
            'move_count': final_info['move_count'],
            'game_state': final_info['game_state'],
            'moves_made': moves_made
        }


class EnhancedExpectimaxAgent:
    """
    Enhanced Expectimax agent that integrates Actor-Critic neural networks.
    Uses Critic for leaf evaluation and Actor for move ordering/pruning.
    """
    
    def __init__(self, game: Game2048, depth: int = 3, 
                 actor=None, critic=None, integrator=None):
        """
        Initialize the enhanced agent.
        
        Args:
            game: Game2048 instance to play
            depth: Maximum search depth for expectimax algorithm
            actor: Actor network (optional, will create if None)
            critic: Critic network (optional, will create if None)
            integrator: ExpectimaxIntegrator (optional, will create if None)
        """
        self.game = game
        self.name = "Enhanced Expectimax Agent (Actor-Critic)"
        self.depth = depth
        

        
        # Initialize Actor-Critic components
        if actor is None or critic is None:
            self.actor, self.critic, self.trainer = create_actor_critic_models()
        else:
            self.actor = actor
            self.critic = critic
            self.trainer = None
        
        self.integrator = integrator or ExpectimaxIntegrator(self.actor, self.critic)
        self.use_neural_evaluation = True
        self.use_move_ordering = True
        self.use_move_pruning = False  # Can be enabled for faster search
    
    def get_move(self) -> Optional[Direction]:
        """
        Get the next move using enhanced expectimax algorithm.
        
        Returns:
            Optional[Direction]: Next move or None if no moves available
        """
        valid_moves = self.game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Use actor for move ordering if enabled
        if self.use_move_ordering:
            ordered_moves = self.integrator.get_ordered_moves(
                self.game, top_k=None if not self.use_move_pruning else 2
            )
            moves_to_evaluate = ordered_moves if ordered_moves else valid_moves
        else:
            moves_to_evaluate = valid_moves
        
        best_move = moves_to_evaluate[0]  # Default to first move
        best_score = float('-inf')
        
        for move in moves_to_evaluate:
            # Skip moves with very low actor probability if pruning enabled
            if self.use_move_pruning and self.integrator.should_prune_move(
                self.game, move, prob_threshold=0.1
            ):
                continue
            
            # Create a copy of the game to test the move
            test_game = self._copy_game_state(self.game)
            if test_game.move(move):
                # Use enhanced expectimax to evaluate this move
                score = self._expectimax(test_game, self.depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    
    def _expectimax(self, game: Game2048, depth: int, is_max_node: bool) -> float:
        """
        Enhanced expectimax algorithm with neural network evaluation.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            is_max_node: True for MAX nodes (player moves), False for CHANCE nodes
            
        Returns:
            float: Expected score for this node
        """
        # Terminal conditions
        if depth == 0 or game.get_game_state() != GameState.ONGOING:
            if self.use_neural_evaluation:
                return self.integrator.evaluate_leaf_state(game)
            else:
                return self._evaluate_board_heuristic(game)
        
        if is_max_node:
            # MAX node: Player's turn, maximize score
            max_score = float('-inf')
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                if self.use_neural_evaluation:
                    return self.integrator.evaluate_leaf_state(game)
                else:
                    return self._evaluate_board_heuristic(game)
            
            # Use actor for move ordering if enabled
            if self.use_move_ordering:
                ordered_moves = self.integrator.get_ordered_moves(game)
                moves_to_evaluate = ordered_moves if ordered_moves else valid_moves
            else:
                moves_to_evaluate = valid_moves
            
            for move in moves_to_evaluate:
                test_game = self._copy_game_state(game)
                if test_game.move(move):
                    score = self._expectimax(test_game, depth - 1, False)
                    max_score = max(max_score, score)
            
            return max_score
        else:
            # CHANCE node: Random tile placement, calculate expected value
            empty_cells = [(i, j) for i in range(game.size) 
                          for j in range(game.size) if game.board[i][j] == 0]
            
            if not empty_cells:
                if self.use_neural_evaluation:
                    return self.integrator.evaluate_leaf_state(game)
                else:
                    return self._evaluate_board_heuristic(game)
            
            expected_score = 0.0
            
            # For each empty cell, consider placing a 2 (90%) or 4 (10%)
            for row, col in empty_cells:
                # Place tile with value 2 (90% probability)
                test_game_2 = self._copy_game_state(game)
                test_game_2.board[row][col] = 2
                score_2 = self._expectimax(test_game_2, depth - 1, True)
                
                # Place tile with value 4 (10% probability)
                test_game_4 = self._copy_game_state(game)
                test_game_4.board[row][col] = 4
                score_4 = self._expectimax(test_game_4, depth - 1, True)
                
                # Expected value for this cell
                cell_expected = 0.9 * score_2 + 0.1 * score_4
                expected_score += cell_expected
            
            # Average over all possible placements
            return expected_score / len(empty_cells)
    
    def _evaluate_board_heuristic(self, game: Game2048) -> float:
        """Fallback heuristic evaluation (same as original SimpleAgent)."""
        if game.get_game_state() == GameState.LOST:
            return float('-inf')
        
        board = game.get_board()
        score = 0.0
        
        # Current game score
        score += game.get_score() * 1.0
        
        # Number of empty cells
        empty_cells = sum(1 for i in range(game.size) 
                         for j in range(game.size) if board[i][j] == 0)
        score += empty_cells * 100.0
        
        # Maximum tile value
        max_tile = max(max(row) for row in board)
        score += max_tile * 10.0
        
        return score
    
    def _copy_game_state(self, game: Game2048) -> Game2048:
        """Create a deep copy of the game state."""
        new_game = Game2048(size=game.size)
        new_game.board = [row[:] for row in game.board]
        new_game.score = game.score
        new_game.game_state = game.game_state
        new_game.move_count = game.move_count
        return new_game
    def _compute_shaped_reward(
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
    
    def play_game(self, max_moves: int = 1000, display_moves: bool = True, 
                  data_collector=None) -> Dict:
        """
        Play a complete game using the enhanced agent's strategy.
        
        Args:
            max_moves: Maximum number of moves to play
            display_moves: Whether to display each move
            data_collector: Optional GameDataCollector to collect experiences during play
            
        Returns:
            Dict: Game statistics including move sequence for training
        """
        display = Display2048(use_colors=True)
        moves_made = 0
        move_sequence = []
        experiences_collected = 0
        
        if display_moves:
            print(f"Starting game with {self.name}")
            print("=" * 50)
            display.print_board(self.game)
            time.sleep(1)
        
        while (self.game.get_game_state() == GameState.ONGOING and 
               moves_made < max_moves):
            
            # Save current state before making move
            if data_collector:
                current_state = [row[:] for row in self.game.get_board()]
                current_score = self.game.get_score()
                current_max_tile = self.game.get_max_tile()
            
            move = self.get_move()
            
            if move is None:
                break
            

            success = self.game.move(move)
            
            # Collect experience for both successful and failed moves
            if data_collector:
                new_max_tile = self.game.get_max_tile()
                new_state = [row[:] for row in self.game.get_board()]
                new_score = self.game.get_score()
                done = self.game.get_game_state() != GameState.ONGOING
                data_collector.collect_experience(current_state, new_state, current_score, new_score, current_max_tile, new_max_tile, success, done, move)
                experiences_collected += 1
            
            if success:
                moves_made += 1
                move_sequence.append(move)
                
                if display_moves:
                    print(f"\nMove {moves_made}: {move.value}")
                    display.print_board(self.game)
                    time.sleep(0.5)
            else:
                break
        
        # Game finished
        final_info = self.game.get_game_info()
        
        if display_moves:
            print("\n" + "=" * 50)
            print("GAME FINISHED!")
            print(f"Final Score: {final_info['score']}")
            print(f"Max Tile: {final_info['max_tile']}")
            print(f"Total Moves: {final_info['move_count']}")
            print(f"Game State: {final_info['game_state'].value}")
        
        return {
            'agent_name': self.name,
            'final_score': final_info['score'],
            'max_tile': final_info['max_tile'],
            'move_count': final_info['move_count'],
            'game_state': final_info['game_state'],
            'moves_made': moves_made,
            'move_sequence': move_sequence,  # For training data collection
            'experiences_collected': experiences_collected  # Number of experiences collected
        }


def train_actor_critic_agent(num_episodes: int = 100, display_games: bool = False,
                           save_models_path: Optional[str] = None) -> Dict:
    """
    Train an Actor-Critic enhanced Expectimax agent.
    
    Args:
        num_episodes: Number of complete games to play for training
        display_games: Whether to display games during training
        save_models_path: Path prefix to save trained models (optional)
        
    Returns:
        Dict: Training statistics and final performance
    """
    # Use save path from config if not provided
    save_models_path = save_models_path or Config.ensure_model_directory()
    
    print(f"=== TRAINING ACTOR-CRITIC ENHANCED EXPECTIMAX AGENT ===")
    print(f"Episodes to train: {num_episodes}")
    print(f"Model save path: {save_models_path}")
    print("=" * 60)
    
    # Create models and trainer with reduced critic learning rate for stability
    actor, critic, trainer = create_actor_critic_models(
        hidden_size=512,
        actor_lr=1e-4,    # Keep actor LR the same
        critic_lr=1e-5    
    )
    data_collector = GameDataCollector(trainer)
    
    # Training statistics
    episode_scores = []
    episode_max_tiles = []
    training_stats = {
        'actor_losses': [],
        'critic_losses': [],
        'actor_entropies': []
    }
    
    # Performance tracking for early stopping
    best_avg_score = 0
    patience_counter = 0
    patience_limit = 50  # Episodes without improvement before adjusting
    
    for episode in range(num_episodes):
        # Create a fresh game for this episode
        game = Game2048()
        
        # Create enhanced agent with current models
        integrator = ExpectimaxIntegrator(actor, critic)
        agent = EnhancedExpectimaxAgent(game, depth=3, actor=actor, 
                                       critic=critic, integrator=integrator)
        
        # Play one complete game and collect experiences during actual gameplay
        display_this_game = display_games and (episode % 20 == 0)  # Show every 20th game
        result = agent.play_game(max_moves=500, display_moves=display_this_game, 
                                data_collector=data_collector)
        
        # Record episode statistics
        episode_scores.append(result['final_score'])
        episode_max_tiles.append(result['max_tile'])
        
        # Training phase: Multiple training steps per episode
        if len(trainer.experience_buffer) >= 32:
            episode_actor_losses = []
            episode_critic_losses = []
            episode_actor_entropies = []

            # Train multiple times on different random batches
            training_steps_per_episode = min(10, len(trainer.experience_buffer) // 32)
            for _ in range(training_steps_per_episode):
                actor_loss, critic_loss, actor_entropy = trainer.train_step(batch_size=32)
                episode_actor_losses.append(actor_loss)
                episode_critic_losses.append(critic_loss)
                episode_actor_entropies.append(actor_entropy)
            
            # Record average losses for this episode
            if episode_actor_losses:
                training_stats['actor_losses'].append(sum(episode_actor_losses) / len(episode_actor_losses))
                training_stats['critic_losses'].append(sum(episode_critic_losses) / len(episode_critic_losses))
                training_stats['actor_entropies'].append(sum(episode_actor_entropies) / len(episode_actor_entropies))
        
        # Performance monitoring and adaptation
        if episode % 10 == 0 and episode > 20:
            recent_scores = episode_scores[-10:] if len(episode_scores) >= 10 else episode_scores
            avg_score = sum(recent_scores) / len(recent_scores)
            
            # Check for performance improvement
            if avg_score > best_avg_score * 1.05:  # 5% improvement threshold
                best_avg_score = avg_score
                patience_counter = 0
            else:
                patience_counter += 10  # Increment by checking interval
                
                # If no improvement for too long, boost exploration
                if patience_counter >= patience_limit:
                    trainer.current_entropy_coeff = min(0.05, trainer.current_entropy_coeff * 2.0)
                    print(f"🔄 Performance plateau detected! Boosting exploration to {trainer.current_entropy_coeff:.4f}")
                    patience_counter = 0
        
        # Print progress
        if episode % 10 == 0 or episode == num_episodes - 1:
            recent_scores = episode_scores[-10:] if len(episode_scores) >= 10 else episode_scores
            avg_score = sum(recent_scores) / len(recent_scores)
            
            # Get diagnostic info
            training_info = trainer.get_training_info()
            
            loss_info = ""
            if training_stats['actor_losses'] and training_stats['critic_losses']:
                loss_info = (f", A_Loss: {training_stats['actor_losses'][-1]:.4f}, "
                           f"C_Loss: {training_stats['critic_losses'][-1]:.4f}, "
                           f"Entropy: {training_stats['actor_entropies'][-1]:.4f}, "
                           f"Ent_Coeff: {training_info['current_entropy_coeff']:.4f}")
            
            print(f"Episode {episode + 1:4d}: Score {result['final_score']:6d}, "
                  f"Max Tile {result['max_tile']:4d}, "
                  f"Avg Score (last 10): {avg_score:.1f}, "
                  f"Buffer: {training_info['buffer_size']}{loss_info}")
    
    # Training completed
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    
    # Calculate final statistics
    final_stats = {
        'total_episodes': num_episodes,
        'final_avg_score': sum(episode_scores[-20:]) / min(20, len(episode_scores)),
        'best_score': max(episode_scores),
        'final_avg_max_tile': sum(episode_max_tiles[-20:]) / min(20, len(episode_max_tiles)),
        'highest_tile': max(episode_max_tiles),
        'buffer_size': len(trainer.experience_buffer),
        'episode_scores': episode_scores,
        'episode_max_tiles': episode_max_tiles,
        'training_losses': training_stats
    }
    
    print(f"Final Average Score (last 20): {final_stats['final_avg_score']:.1f}")
    print(f"Best Score Achieved: {final_stats['best_score']}")
    print(f"Final Average Max Tile: {final_stats['final_avg_max_tile']:.1f}")
    print(f"Highest Tile Reached: {final_stats['highest_tile']}")
    print(f"Experience Buffer Size: {final_stats['buffer_size']}")
    
    # Save models if path provided
    if save_models_path:
        save_models(actor, critic, save_models_path)
        print(f"Models saved to {save_models_path}_actor.pth and {save_models_path}_critic.pth")
    
    return final_stats


def demo_basic_usage():
    """Demonstrate basic game usage."""
    print("=== BASIC GAME USAGE DEMO ===\n")
    
    # Create a new game
    game = Game2048()
    display = Display2048()
    
    print("1. Creating a new game:")
    print_game(game)
    
    print("\n2. Making some moves:")
    moves = [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN]
    
    for i, move in enumerate(moves):
        print(f"\nMove {i+1}: {move.value}")
        success = game.move(move)
        print(f"Move successful: {success}")
        
        if success:
            print_game(game)
        else:
            print("Move was not possible!")
        
        time.sleep(1)
    
    print("\n3. Game information:")
    info = game.get_game_info()
    print(f"Score: {info['score']}")
    print(f"Max Tile: {info['max_tile']}")
    print(f"Move Count: {info['move_count']}")
    print(f"Valid Moves: {[move.value for move in info['valid_moves']]}")


def demo_agent_interaction():
    """Demonstrate agent interaction with the game."""
    print("\n=== AGENT INTERACTION DEMO ===\n")
    
    # Create games for different agents
    game1 = Game2048()
    game2 = Game2048()
    
    # Create agents
    expectimax_agent = SimpleAgent(game1)
    random_agent = RandomAgent(game2)
    
    print("1. Playing with Expectimax Agent:")
    result1 = expectimax_agent.play_game(max_moves=50, display_moves=True)
    
    print("\n2. Playing with Random Agent (fast):")
    result2 = random_agent.play_game(max_moves=100, display_moves=False)
    
    print("\nAgent Comparison:")
    print(f"Expectimax Agent - Score: {result1['final_score']}, Max Tile: {result1['max_tile']}")
    print(f"Random Agent - Score: {result2['final_score']}, Max Tile: {result2['max_tile']}")


def demo_multiple_games():
    """Run multiple games to show statistics."""
    print("\n=== MULTIPLE GAMES STATISTICS ===\n")
    
    num_games = 5
    results = []
    
    print(f"Running {num_games} games with Random Agent...")
    
    for i in range(num_games):
        game = Game2048()
        agent = RandomAgent(game)
        result = agent.play_game(max_moves=200, display_moves=False)
        results.append(result)
        print(f"Game {i+1}: Score {result['final_score']}, Max Tile {result['max_tile']}")
    
    # Calculate statistics
    scores = [r['final_score'] for r in results]
    max_tiles = [r['max_tile'] for r in results]
    
    print(f"\nStatistics over {num_games} games:")
    print(f"Average Score: {sum(scores) / len(scores):.1f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Average Max Tile: {sum(max_tiles) / len(max_tiles):.1f}")
    print(f"Highest Tile Reached: {max(max_tiles)}")


def demo_manual_play():
    """Demo function for manual play (keyboard input)."""
    print("\n=== MANUAL PLAY DEMO ===\n")
    print("Use WASD keys to move:")
    print("W = Up, A = Left, S = Down, D = Right")
    print("Q = Quit")
    print("-" * 30)
    
    game = Game2048()
    display = Display2048()
    
    key_to_direction = {
        'w': Direction.UP,
        'a': Direction.LEFT,
        's': Direction.DOWN,
        'd': Direction.RIGHT
    }
    
    while game.get_game_state() == GameState.ONGOING:
        print_game(game)
        
        try:
            key = input("\nEnter move (WASD) or Q to quit: ").lower().strip()
            
            if key == 'q':
                print("Thanks for playing!")
                break
            
            if key not in key_to_direction:
                print("Invalid input! Use W, A, S, D, or Q")
                continue
            
            direction = key_to_direction[key]
            success = game.move(direction)
            
            if not success:
                print("Invalid move! Try another direction.")
                continue
                
        except KeyboardInterrupt:
            print("\nGame interrupted!")
            break
    
    # Final game state
    if game.get_game_state() != GameState.ONGOING:
        print_game(game)
        final_info = game.get_game_info()
        print(f"\nFinal Score: {final_info['score']}")
        print(f"Max Tile: {final_info['max_tile']}")


def demo_custom_agent():
    """Show how to create a custom agent."""
    print("\n=== CUSTOM AGENT DEMO ===\n")
    
    class GreedyAgent:
        """Agent that always chooses the move that maximizes immediate score."""
        
        def __init__(self, game: Game2048):
            self.game = game
            self.name = "Greedy Agent"
        
        def get_move(self) -> Optional[Direction]:
            """Get the move that maximizes immediate score gain."""
            valid_moves = self.game.get_valid_moves()
            
            if not valid_moves:
                return None
            
            best_move = None
            best_score_gain = -1
            
            for move in valid_moves:
                # Test the move on a copy of the game
                test_game = Game2048()
                test_game.board = [row[:] for row in self.game.board]
                test_game.score = self.game.score
                
                old_score = test_game.score
                test_game.move(move)
                score_gain = test_game.score - old_score
                
                if score_gain > best_score_gain:
                    best_score_gain = score_gain
                    best_move = move
            
            return best_move or valid_moves[0]
    
    # Test the custom agent
    game = Game2048()
    agent = GreedyAgent(game)
    
    print("Running Greedy Agent that maximizes immediate score...")
    
    moves_made = 0
    while game.get_game_state() == GameState.ONGOING and moves_made < 20:
        move = agent.get_move()
        if move and game.move(move):
            moves_made += 1
            print(f"Move {moves_made}: {move.value}")
            print_game(game)
            time.sleep(0.5)
        else:
            break
    
    print(f"\nGreedy Agent final score: {game.get_score()}")

def main():
    """Main demo function."""
    print("Welcome to the 2048 Game Demo!")
    print("=" * 50)
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Agent Interaction", demo_agent_interaction),
        ("Multiple Games Statistics", demo_multiple_games),
        ("Custom Agent", demo_custom_agent),
        ("Manual Play", demo_manual_play),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    
    print("\nRunning all demos (except manual play)...")
    
    for name, demo_func in demos[:-1]:  # Skip manual play in auto mode
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"\n{name} interrupted!")
            break
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    print("\n" + "="*50)
    print("Demo completed!")
    print("To play manually, run: python demo.py manual")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "manual":
            demo_manual_play()
        elif sys.argv[1] == "agent":
            agent = SimpleAgent(Game2048())
            agent.play_game(max_moves=200, display_moves=True)
        elif sys.argv[1] == "train":
            # Train Actor-Critic agent with more episodes
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            train_actor_critic_agent(num_episodes=episodes, display_games=False)
        elif sys.argv[1] == "enhanced":
            # Play with enhanced agent (requires trained models)
            try:
                actor, critic = load_models("demo_models")
                game = Game2048()
                integrator = ExpectimaxIntegrator(actor, critic)
                agent = EnhancedExpectimaxAgent(game, depth=3, actor=actor, 
                                               critic=critic, integrator=integrator)
                agent.play_game(max_moves=500, display_moves=True)
            except Exception as e:
                print(f"Error: {e}")
                print("Train models first with: python demo.py train")
        else:
            print("Usage:")
            print("  python demo.py          - Run all demos")
            print("  python demo.py manual   - Manual play mode")
            print("  python demo.py agent    - Simple Expectimax agent")
            print("  python demo.py train [episodes] - Train Actor-Critic (default: 100)")
            print("  python demo.py enhanced - Play with trained enhanced agent")
    else:
        main() 