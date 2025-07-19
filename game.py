import random
import copy
from typing import List, Tuple, Optional, Dict
from enum import Enum


class Direction(Enum):
    """Enum for movement directions"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class GameState(Enum):
    """Enum for game states"""
    ONGOING = "ongoing"
    WON = "won"
    LOST = "lost"


class Game2048:
    """
    A complete 2048 game implementation with authentic mechanics and probabilities.
    Designed to support both human players and AI agents.
    """
    
    def __init__(self, size: int = 4):
        """
        Initialize a new 2048 game.
        
        Args:
            size: Size of the game board (default 4x4)
        """
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.score = 0
        self.game_state = GameState.ONGOING
        self.move_count = 0
        self.max_tile = 0
        
        # Add two initial tiles
        self.add_random_tile()
        self.add_random_tile()
    
    def add_random_tile(self) -> bool:
        """
        Add a random tile to the board with authentic 2048 probabilities.
        90% chance for 2, 10% chance for 4.
        
        Returns:
            bool: True if a tile was added, False if board is full
        """
        empty_cells = [(i, j) for i in range(self.size) 
                      for j in range(self.size) if self.board[i][j] == 0]
        
        if not empty_cells:
            return False
        
        # Choose random empty cell
        row, col = random.choice(empty_cells)
        
        # 90% chance for 2, 10% chance for 4 (authentic 2048 probabilities)
        value = 2 if random.random() < 0.9 else 4
        self.board[row][col] = value
        
        return True
    
    def get_board(self) -> List[List[int]]:
        """
        Get a copy of the current board state.
        
        Returns:
            List[List[int]]: Copy of the board
        """
        return copy.deepcopy(self.board)
    
    def get_score(self) -> int:
        """Get the current score."""
        return self.score
    
    def get_game_state(self) -> GameState:
        """Get the current game state."""
        return self.game_state
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value on the board."""
        return max(max(row) for row in self.board)
    
    def _slide_row_left(self, row: List[int]) -> Tuple[List[int], int]:
        """
        Slide a row to the left and merge tiles.
        
        Args:
            row: List representing a row
            
        Returns:
            Tuple[List[int], int]: (new_row, points_gained)
        """
        # Remove zeros and slide left
        non_zero = [x for x in row if x != 0]
        
        # Merge adjacent identical tiles
        merged = []
        points = 0
        i = 0
        
        while i < len(non_zero):
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
                # Merge tiles
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                points += merged_value
                i += 2  # Skip the next tile as it's been merged
            else:
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros to maintain row length
        merged.extend([0] * (self.size - len(merged)))
        
        return merged, points
    
    def _slide_row_right(self, row: List[int]) -> Tuple[List[int], int]:
        """
        Slide a row to the right and merge tiles.
        
        Args:
            row: List representing a row
            
        Returns:
            Tuple[List[int], int]: (new_row, points_gained)
        """
        # Reverse, slide left, then reverse back
        reversed_row = row[::-1]
        merged, points = self._slide_row_left(reversed_row)
        return merged[::-1], points
    
    def _transpose_board(self) -> None:
        """Transpose the board (swap rows and columns)."""
        self.board = [[self.board[j][i] for j in range(self.size)] 
                     for i in range(self.size)]
    
    def _reverse_rows(self) -> None:
        """Reverse each row of the board."""
        for i in range(self.size):
            self.board[i] = self.board[i][::-1]
    
    def move(self, direction: Direction) -> bool:
        """
        Make a move in the specified direction.
        
        Args:
            direction: Direction to move
            
        Returns:
            bool: True if the move was valid and changed the board, False otherwise
        """
        if self.game_state != GameState.ONGOING:
            return False
        
        # Store original board state
        original_board = copy.deepcopy(self.board)
        points_gained = 0
        
        if direction == Direction.LEFT:
            for i in range(self.size):
                self.board[i], row_points = self._slide_row_left(self.board[i])
                points_gained += row_points
                
        elif direction == Direction.RIGHT:
            for i in range(self.size):
                self.board[i], row_points = self._slide_row_right(self.board[i])
                points_gained += row_points
                
        elif direction == Direction.UP:
            self._transpose_board()
            for i in range(self.size):
                self.board[i], row_points = self._slide_row_left(self.board[i])
                points_gained += row_points
            self._transpose_board()
            
        elif direction == Direction.DOWN:
            self._transpose_board()
            for i in range(self.size):
                self.board[i], row_points = self._slide_row_right(self.board[i])
                points_gained += row_points
            self._transpose_board()
        
        # Check if the move changed the board
        if self.board == original_board:
            return False
        
        # Update score and move count
        self.score += points_gained
        self.move_count += 1
        
        # Add new random tile
        self.add_random_tile()
        
        # Update game state
        self._update_game_state()
        
        return True
    
    def _update_game_state(self) -> None:
        """Update the game state (ongoing, won, lost)."""
        # Check for 2048 tile (win condition)
        max_tile = self.get_max_tile()
        if max_tile >= 2048 and self.game_state == GameState.ONGOING:
            self.game_state = GameState.WON
            # Note: In authentic 2048, you can continue playing after winning
            # so we don't stop the game here
        
        # Check for game over
        if self._is_game_over():
            self.game_state = GameState.LOST
    
    def _is_game_over(self) -> bool:
        """
        Check if the game is over (no more moves possible).
        
        Returns:
            bool: True if game is over, False otherwise
        """
        # Check for empty cells
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return False
        
        # Check for possible merges horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i][j] == self.board[i][j + 1]:
                    return False
        
        # Check for possible merges vertically
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i][j] == self.board[i + 1][j]:
                    return False
        
        return True
    
    def get_valid_moves(self) -> List[Direction]:
        """
        Get all valid moves from the current state.
        
        Returns:
            List[Direction]: List of valid directions
        """
        valid_moves = []
        
        for direction in Direction:
            # Create a temporary game state to test the move
            temp_game = copy.deepcopy(self)
            if temp_game.move(direction):
                valid_moves.append(direction)
        
        return valid_moves
    
    def is_move_valid(self, direction: Direction) -> bool:
        """
        Check if a move in the given direction is valid.
        
        Args:
            direction: Direction to check
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        return direction in self.get_valid_moves()
    
    def reset(self) -> None:
        """Reset the game to initial state."""
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0
        self.game_state = GameState.ONGOING
        self.move_count = 0
        
        # Add two initial tiles
        self.add_random_tile()
        self.add_random_tile()
    
    def get_game_info(self) -> Dict:
        """
        Get comprehensive game information for agents.
        
        Returns:
            Dict: Dictionary containing game state information
        """
        return {
            'board': self.get_board(),
            'score': self.score,
            'game_state': self.game_state,
            'move_count': self.move_count,
            'max_tile': self.get_max_tile(),
            'valid_moves': self.get_valid_moves(),
            'is_game_over': self.game_state == GameState.LOST
        }
    
    def __str__(self) -> str:
        """String representation of the game board."""
        lines = []
        for row in self.board:
            line = "|"
            for cell in row:
                if cell == 0:
                    line += "    |"
                else:
                    line += f"{cell:4d}|"
            lines.append(line)
        
        return "\n".join(lines) 