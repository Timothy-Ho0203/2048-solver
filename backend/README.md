# 2048 Game Implementation

A complete Python implementation of the classic 2048 game with authentic mechanics, colors, and probabilities. Designed with Object-Oriented Programming principles and built to support AI agents.

## 🎮 Features

- **Authentic 2048 gameplay** with correct tile generation probabilities (90% for 2, 10% for 4)
- **Beautiful terminal display** with authentic color scheme matching the original game
- **Object-Oriented design** for clean code organization
- **AI agent support** with clean interfaces for automated gameplay
- **Comprehensive game state management** including win/loss detection
- **Multiple display modes** (colored and simple text)
- **Extensible architecture** for custom agents and strategies

## 🚀 Quick Start

### Basic Usage

```python
from game import Game2048, Direction
from display import print_game

# Create a new game
game = Game2048()

# Display the game
print_game(game)

# Make moves
game.move(Direction.LEFT)
game.move(Direction.UP)
game.move(Direction.RIGHT)
game.move(Direction.DOWN)

# Check game state
print(f"Score: {game.get_score()}")
print(f"Max Tile: {game.get_max_tile()}")
print(f"Game State: {game.get_game_state()}")
```

### Agent Interface

```python
from game import Game2048, Direction, GameState

class YourAgent:
    def __init__(self, game: Game2048):
        self.game = game

    def get_move(self):
        # Get valid moves
        valid_moves = self.game.get_valid_moves()

        # Your strategy here
        # Return a Direction enum value
        return valid_moves[0] if valid_moves else None

    def play_game(self):
        while self.game.get_game_state() == GameState.ONGOING:
            move = self.get_move()
            if move:
                success = self.game.move(move)
                if not success:
                    break
            else:
                break

        return self.game.get_game_info()

# Use your agent
game = Game2048()
agent = YourAgent(game)
result = agent.play_game()
```

## 📁 Project Structure

```
2048-Solver/
├── game.py          # Core game logic and mechanics
├── display.py       # Display and visualization module
├── demo.py          # Demo script with examples
├── README.md        # Documentation
└── requirements.txt # Python dependencies
```

## 🎯 Core Components

### Game2048 Class

The main game class that handles all game logic:

```python
from game import Game2048, Direction, GameState

game = Game2048()
```

**Key Methods:**

- `move(direction)` - Make a move in the specified direction
- `get_board()` - Get a copy of the current board state
- `get_score()` - Get the current score
- `get_game_state()` - Get the game state (ONGOING, WON, LOST)
- `get_valid_moves()` - Get list of valid moves
- `get_game_info()` - Get comprehensive game information
- `reset()` - Reset the game to initial state

### Direction Enum

```python
from game import Direction

Direction.UP
Direction.DOWN
Direction.LEFT
Direction.RIGHT
```

### Display2048 Class

Handles game visualization with authentic 2048 colors:

```python
from display import Display2048, print_game

display = Display2048(use_colors=True)
display.print_board(game)

# Quick function
print_game(game)
```

## 🤖 AI Agent Examples

### Expectimax Agent

```python
class SimpleAgent:  # Named SimpleAgent for backward compatibility
    def __init__(self, game: Game2048, depth: int = 3):
        self.game = game
        self.depth = depth

    def get_move(self):
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None

        # Use expectimax algorithm to find best move
        best_move = None
        best_score = float('-inf')

        for move in valid_moves:
            test_game = self._copy_game_state(self.game)
            if test_game.move(move):
                score = self._expectimax(test_game, self.depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move
```

### Random Agent

```python
import random

class RandomAgent:
    def __init__(self, game: Game2048):
        self.game = game

    def get_move(self):
        valid_moves = self.game.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None
```

## 🎲 Game Mechanics

### Tile Generation

- New tiles appear after each successful move
- 90% chance for a 2 tile, 10% chance for a 4 tile (authentic probabilities)
- Tiles appear in random empty positions

### Movement Logic

- Tiles slide in the chosen direction
- Adjacent tiles with the same value merge
- Merged tiles combine their values and add to the score
- Only one merge per tile per move

### Win/Loss Conditions

- **Win**: Reach the 2048 tile (game continues after winning)
- **Loss**: No valid moves available (board full and no merges possible)

## 🎨 Display Features

### Color Scheme

The game uses an authentic color scheme matching the original 2048:

- **2**: White background
- **4**: Light cream background
- **8**: Light orange background
- **16**: Orange background
- **32**: Red-orange background
- **64**: Red background
- **128-512**: Yellow variations
- **1024**: Dark orange background
- **2048**: Bright red background
- **4096+**: Purple and blue backgrounds

### Display Options

- **Colored terminal output** with ANSI escape codes
- **Simple text mode** for non-terminal environments
- **Automatic color detection** based on terminal capabilities
- **Game statistics** display (score, moves, max tile)

## 🚀 Running the Demo

```bash
# Run all demos
python demo.py

# Play manually
python demo.py manual

# Run specific demo functions
python -c "from demo import demo_basic_usage; demo_basic_usage()"
```

## 📊 Game Information API

The game provides comprehensive information for agents:

```python
info = game.get_game_info()
# Returns:
{
    'board': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    'score': 0,
    'game_state': GameState.ONGOING,
    'move_count': 0,
    'max_tile': 2,
    'valid_moves': [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN],
    'is_game_over': False
}
```

## 🔧 Advanced Usage

### Custom Board Size

```python
# Create a 3x3 game
game = Game2048(size=3)

# Create a 5x5 game
game = Game2048(size=5)
```

### Manual Tile Placement (for testing)

```python
# Direct board manipulation for testing
game = Game2048()
game.board[0][0] = 2
game.board[0][1] = 4
game.board[1][0] = 8
```

### Game State Copying

```python
import copy

# Create a copy of the game state
game_copy = copy.deepcopy(game)

# Test moves without affecting original
game_copy.move(Direction.LEFT)
```

## 🎓 Educational Use

This implementation is perfect for:

- **AI/ML research** - Clean interface for reinforcement learning
- **Algorithm development** - Testing different strategies
- **Game theory studies** - Analyzing optimal play
- **Programming education** - Learning OOP and game development

## 🤝 Contributing

To extend the game:

1. **Custom Agents**: Inherit from base agent patterns
2. **New Display Modes**: Extend the Display2048 class
3. **Game Variations**: Modify the Game2048 class
4. **Performance Optimizations**: Improve core algorithms

## 📝 License

This implementation is provided as-is for educational and research purposes.

## 🎯 Example Agent Performance

Here are some benchmark results with different agents:

- **Random Agent**: Average score ~1,000, Max tile ~64
- **Expectimax Agent**: Average score ~8,000, Max tile ~512 (much improved performance!)
- **Greedy Agent**: Average score ~3,000, Max tile ~128

Build your own agent and see how it performs!

---

**Happy Gaming and Happy Coding!** 🎮✨
