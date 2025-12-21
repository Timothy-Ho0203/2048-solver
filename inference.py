#!/usr/bin/env python3
"""
Inference module for using trained Actor model during 2048 gameplay.
Provides a simple interface to get optimal moves from trained models.
"""

import torch
from typing import Optional, Dict
from game import Game2048, Direction
from actor_critic import load_models
from config import Config


class GameInferenceEngine:
    """Lightweight inference engine for 2048 gameplay using Actor network."""

    def __init__(self, model_path_prefix: str = None):
        """
        Load trained models for inference.

        Args:
            model_path_prefix: Path prefix for model files.
                             Defaults to Config.MODEL_SAVE_PATH
        """
        if model_path_prefix is None:
            model_path_prefix = Config.MODEL_SAVE_PATH

        print(f"Loading models from {model_path_prefix}...")
        self.actor, self.critic = load_models(model_path_prefix)

        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()

        print("Models loaded successfully!")

    def get_best_move(self, game: Game2048) -> Optional[Direction]:
        """
        Get the best move using the actor network.

        Args:
            game: Current game state

        Returns:
            Direction: Optimal next move, or None if no valid moves
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        # Get action probabilities from actor
        action_probs = self.actor.get_action_probabilities(
            game.get_board(),
            valid_moves
        )

        # Return move with highest probability
        best_move = max(action_probs.items(), key=lambda x: x[1])[0]
        return best_move

    def get_move_with_probabilities(self, game: Game2048) -> Dict:
        """
        Get the best move along with all action probabilities.
        Always returns probabilities for all 4 directions (invalid moves have 0.0).

        Args:
            game: Current game state

        Returns:
            Dict: Contains 'best_move', 'confidence', and 'all_probabilities'
        """
        # Initialize all directions with 0.0 probability
        all_probs = {
            'up': 0.0,
            'down': 0.0,
            'left': 0.0,
            'right': 0.0
        }

        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return {
                'best_move': None,
                'confidence': 0.0,
                'all_probabilities': all_probs
            }

        # Get action probabilities from actor (only for valid moves)
        action_probs = self.actor.get_action_probabilities(
            game.get_board(),
            valid_moves
        )

        # Find best move
        best_move = max(action_probs.items(), key=lambda x: x[1])[0]
        confidence = action_probs[best_move]

        # Update probabilities for valid moves
        for move, prob in action_probs.items():
            all_probs[move.value] = prob

        return {
            'best_move': best_move.value,
            'confidence': confidence,
            'all_probabilities': all_probs
        }

    def get_move_from_board(self, board: list) -> Dict:
        """
        Get the best move directly from a board state.
        Convenience method for API usage.

        Args:
            board: 4x4 board state (list of lists)

        Returns:
            Dict: Contains 'best_move', 'confidence', and 'all_probabilities'
        """
        # Create a game instance with the given board
        game = Game2048()
        game.board = [row[:] for row in board]  # Deep copy

        return self.get_move_with_probabilities(game)


# Example usage
if __name__ == "__main__":
    print("=== Testing Game Inference Engine ===\n")

    # Initialize engine
    engine = GameInferenceEngine()

    # Create a test game
    game = Game2048()
    print("Initial board:")
    from display import print_game
    print_game(game)

    # Get best move
    result = engine.get_move_with_probabilities(game)

    print(f"\nBest move: {result['best_move']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll move probabilities:")
    for move, prob in result['all_probabilities'].items():
        print(f"  {move}: {prob:.2%}")

    # Test playing a few moves
    print("\n=== Playing 5 moves with AI ===")
    for i in range(5):
        move = engine.get_best_move(game)
        if move is None:
            print("No valid moves!")
            break

        success = game.move(move)
        if success:
            print(f"\nMove {i+1}: {move.value}")
            print_game(game)
        else:
            print(f"Move {move.value} failed!")
            break

    print(f"\nFinal Score: {game.get_score()}")
    print(f"Max Tile: {game.get_max_tile()}")
