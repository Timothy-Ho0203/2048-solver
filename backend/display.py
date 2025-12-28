import os
import sys
from typing import List, Dict, Optional
from game import Game2048, GameState


class Display2048:
    """
    Display module for 2048 game with authentic colors and styling.
    Provides both terminal and formatted output for the game board.
    """
    
    # Authentic 2048 color scheme (using ANSI color codes)
    TILE_COLORS = {
        0: "\033[48;5;250m\033[30m",      # Light gray background, black text
        2: "\033[48;5;255m\033[30m",      # White background, black text
        4: "\033[48;5;230m\033[30m",      # Light cream background, black text
        8: "\033[48;5;215m\033[30m",      # Light orange background, black text
        16: "\033[48;5;209m\033[97m",     # Orange background, white text
        32: "\033[48;5;203m\033[97m",     # Red-orange background, white text
        64: "\033[48;5;196m\033[97m",     # Red background, white text
        128: "\033[48;5;220m\033[30m",    # Light yellow background, black text
        256: "\033[48;5;214m\033[30m",    # Yellow background, black text
        512: "\033[48;5;208m\033[30m",    # Dark yellow background, black text
        1024: "\033[48;5;202m\033[97m",   # Dark orange background, white text
        2048: "\033[48;5;196m\033[97m",   # Bright red background, white text
        4096: "\033[48;5;57m\033[97m",    # Purple background, white text
        8192: "\033[48;5;21m\033[97m",    # Blue background, white text
    }
    
    # Reset color
    RESET_COLOR = "\033[0m"
    
    # Board styling
    BORDER_COLOR = "\033[38;5;8m"  # Dark gray
    SCORE_COLOR = "\033[38;5;33m"  # Blue
    TITLE_COLOR = "\033[38;5;226m" # Bright yellow
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the display.
        
        Args:
            use_colors: Whether to use ANSI colors (disable for non-terminal output)
        """
        self.use_colors = use_colors and self._supports_color()
    
    def _supports_color(self) -> bool:
        """
        Check if the terminal supports ANSI colors.
        
        Returns:
            bool: True if colors are supported
        """
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _get_tile_color(self, value: int) -> str:
        """
        Get the color code for a tile value.
        
        Args:
            value: Tile value
            
        Returns:
            str: ANSI color code
        """
        if not self.use_colors:
            return ""
        
        # Use the specific color for the value, or default to highest available
        if value in self.TILE_COLORS:
            return self.TILE_COLORS[value]
        elif value > 8192:
            # For values higher than 8192, use the 8192 color
            return self.TILE_COLORS[8192]
        else:
            return self.TILE_COLORS[0]
    
    def _format_tile(self, value: int, width: int = 6) -> str:
        """
        Format a tile with appropriate colors and padding.
        
        Args:
            value: Tile value
            width: Width of the tile display
            
        Returns:
            str: Formatted tile string
        """
        color = self._get_tile_color(value)
        reset = self.RESET_COLOR if self.use_colors else ""
        
        if value == 0:
            display_value = ""
        else:
            display_value = str(value)
        
        # Center the value in the tile
        padded_value = display_value.center(width)
        
        return f"{color}{padded_value}{reset}"
    
    def draw_board(self, game: Game2048) -> str:
        """
        Draw the game board with colors and styling.
        
        Args:
            game: Game2048 instance
            
        Returns:
            str: Formatted board string
        """
        board = game.get_board()
        lines = []
        
        # Title
        title_color = self.TITLE_COLOR if self.use_colors else ""
        reset = self.RESET_COLOR if self.use_colors else ""
        lines.append(f"{title_color}╔══════════════════════════════╗{reset}")
        lines.append(f"{title_color}║            2048              ║{reset}")
        lines.append(f"{title_color}╚══════════════════════════════╝{reset}")
        lines.append("")
        
        # Score display
        score_color = self.SCORE_COLOR if self.use_colors else ""
        lines.append(f"{score_color}Score: {game.get_score()}{reset}")
        lines.append(f"{score_color}Move Count: {game.move_count}{reset}")
        lines.append(f"{score_color}Max Tile: {game.get_max_tile()}{reset}")
        lines.append("")
        
        # Board border
        border_color = self.BORDER_COLOR if self.use_colors else ""
        
        # Top border
        lines.append(f"{border_color}┌{'─' * 6}┬{'─' * 6}┬{'─' * 6}┬{'─' * 6}┐{reset}")
        
        # Board rows
        for i, row in enumerate(board):
            # Row content
            row_str = f"{border_color}│{reset}"
            for cell in row:
                row_str += self._format_tile(cell)
                row_str += f"{border_color}│{reset}"
            lines.append(row_str)
            
            # Row separator (except for last row)
            if i < len(board) - 1:
                lines.append(f"{border_color}├{'─' * 6}┼{'─' * 6}┼{'─' * 6}┼{'─' * 6}┤{reset}")
        
        # Bottom border
        lines.append(f"{border_color}└{'─' * 6}┴{'─' * 6}┴{'─' * 6}┴{'─' * 6}┘{reset}")
        
        # Game state
        if game.get_game_state() == GameState.WON:
            lines.append(f"\n{self.TITLE_COLOR}🎉 You Won! You reached 2048! 🎉{reset}")
        elif game.get_game_state() == GameState.LOST:
            lines.append(f"\n{self.BORDER_COLOR}💀 Game Over! No more moves possible. 💀{reset}")
        
        return "\n".join(lines)
    
    def print_board(self, game: Game2048) -> None:
        """
        Print the game board to the terminal.
        
        Args:
            game: Game2048 instance
        """
        # Clear screen (optional)
        if self.use_colors:
            os.system('clear' if os.name == 'posix' else 'cls')
        
        print(self.draw_board(game))
    
    def draw_simple_board(self, game: Game2048) -> str:
        """
        Draw a simple text-only board without colors.
        
        Args:
            game: Game2048 instance
            
        Returns:
            str: Simple board string
        """
        board = game.get_board()
        lines = []
        
        lines.append("2048 Game")
        lines.append("=" * 25)
        lines.append(f"Score: {game.get_score()}")
        lines.append(f"Move Count: {game.move_count}")
        lines.append("")
        
        # Simple board representation
        for row in board:
            line = "|"
            for cell in row:
                if cell == 0:
                    line += "    |"
                else:
                    line += f"{cell:4d}|"
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_color_legend(self) -> str:
        """
        Get a legend showing colors for different tile values.
        
        Returns:
            str: Color legend string
        """
        if not self.use_colors:
            return "Color display is disabled."
        
        lines = []
        lines.append("Color Legend:")
        lines.append("=" * 20)
        
        # Show colors for common tile values
        common_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        for value in common_values:
            if value in self.TILE_COLORS:
                color = self.TILE_COLORS[value]
                reset = self.RESET_COLOR
                lines.append(f"{color}{value:4d}{reset} - {value}")
        
        return "\n".join(lines)
    
    def format_game_info(self, game: Game2048) -> str:
        """
        Format comprehensive game information.
        
        Args:
            game: Game2048 instance
            
        Returns:
            str: Formatted game information
        """
        info = game.get_game_info()
        
        lines = []
        lines.append("Game Information:")
        lines.append("=" * 30)
        lines.append(f"Score: {info['score']}")
        lines.append(f"Move Count: {info['move_count']}")
        lines.append(f"Max Tile: {info['max_tile']}")
        lines.append(f"Game State: {info['game_state'].value}")
        lines.append(f"Valid Moves: {[move.value for move in info['valid_moves']]}")
        lines.append(f"Is Game Over: {info['is_game_over']}")
        
        return "\n".join(lines)


# Convenience functions for quick display
def print_game(game: Game2048, use_colors: bool = True) -> None:
    """
    Quick function to print a game board.
    
    Args:
        game: Game2048 instance
        use_colors: Whether to use colors
    """
    display = Display2048(use_colors=use_colors)
    display.print_board(game)


def get_board_string(game: Game2048, use_colors: bool = True) -> str:
    """
    Quick function to get a formatted board string.
    
    Args:
        game: Game2048 instance
        use_colors: Whether to use colors
        
    Returns:
        str: Formatted board string
    """
    display = Display2048(use_colors=use_colors)
    return display.draw_board(game) 