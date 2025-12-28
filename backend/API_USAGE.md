# 2048 AI Move API - Usage Guide

## Overview
This API provides optimal move predictions for the 2048 game using trained Actor-Critic neural networks. It uses only the **Actor network** for fast, real-time inference.

## Quick Start

### 1. Install Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python api.py
```

The server will start on `http://localhost:8000`

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Get optimal move
curl -X POST "http://localhost:8000/get_move" \
  -H "Content-Type: application/json" \
  -d '{"board": [[2,0,0,0],[0,0,0,0],[0,0,0,0],[4,0,0,0]]}'
```

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API and models are loaded successfully.

**Response:**
```json
{"status": "healthy"}
```

---

### 2. Get Move (Simple)
**POST** `/get_move`

Get the optimal move for a given board state.

**Request Body:**
```json
{
  "board": [
    [2, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [4, 0, 0, 0]
  ]
}
```

**Response:**
```json
{
  "best_move": "up",
  "confidence": 1.0,
  "all_probabilities": {
    "up": 1.0,
    "down": 0.0,
    "right": 0.0
  }
}
```

**Fields:**
- `best_move`: Direction to move (up/down/left/right) or null if no valid moves
- `confidence`: Probability of the best move (0.0 to 1.0)
- `all_probabilities`: Probabilities for all valid moves

---

### 3. Get Move (Detailed)
**POST** `/get_move_detailed`

Get the optimal move with additional game information.

**Request Body:** Same as `/get_move`

**Response:**
```json
{
  "best_move": "up",
  "confidence": 1.0,
  "all_probabilities": {
    "up": 1.0,
    "down": 0.0,
    "right": 0.0
  },
  "game_info": {
    "score": 0,
    "max_tile": 1024,
    "valid_moves": ["up", "down", "right"]
  }
}
```

---

### 4. Batch Processing
**POST** `/get_moves_batch`

Get optimal moves for multiple board states in one request.

**Request Body:**
```json
[
  {
    "board": [[2,0,0,0],[0,0,0,0],[0,0,0,0],[4,0,0,0]]
  },
  {
    "board": [[2,4,8,16],[32,64,128,256],[512,1024,2,4],[2,4,2,0]]
  }
]
```

**Response:** Array of move responses

---

## Interactive Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/get_move"

# Board state (4x4 grid)
board = [
    [2, 4, 8, 16],
    [32, 64, 128, 256],
    [512, 1024, 2, 4],
    [2, 4, 2, 0]
]

# Make request
response = requests.post(url, json={"board": board})
result = response.json()

print(f"Best move: {result['best_move']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## JavaScript/TypeScript Example

```javascript
const API_URL = 'http://localhost:8000/get_move';

async function getOptimalMove(board) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ board }),
  });

  const result = await response.json();
  console.log(`Best move: ${result.best_move}`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);

  return result.best_move;
}

// Example usage
const board = [
  [2, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [4, 0, 0, 0]
];

getOptimalMove(board);
```

## Using the Inference Engine Directly (Python)

If you don't need an API and just want to use the inference engine in Python:

```python
from inference import GameInferenceEngine
from game import Game2048

# Initialize engine
engine = GameInferenceEngine()

# Create a game
game = Game2048()

# Get best move
move = engine.get_best_move(game)
print(f"Best move: {move.value}")

# Get move with probabilities
result = engine.get_move_with_probabilities(game)
print(f"Best move: {result['best_move']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['all_probabilities']}")
```

## Board State Format

The board is represented as a 4x4 grid:
- `0` = empty cell
- `2, 4, 8, 16, ...` = tile values (powers of 2)

Example:
```
[[2, 0, 0, 0],     ┌──────┬──────┬──────┬──────┐
 [0, 0, 0, 0],  => │  2   │      │      │      │
 [0, 0, 0, 0],     │      │      │      │      │
 [4, 0, 0, 0]]     │      │      │      │      │
                   │  4   │      │      │      │
                   └──────┴──────┴──────┴──────┘
```

## Performance

- **Response time**: ~5-20ms per request
- **Model size**: ~2MB (actor) + ~8MB (critic)
- **Memory usage**: ~100MB when loaded

## Error Handling

**Invalid board dimensions:**
```json
{
  "detail": "Board must have exactly 4 rows"
}
```

**Models not loaded:**
```json
{
  "detail": "Models not loaded"
}
```

## Production Deployment

For production use, consider:

1. **Change CORS settings** in `api.py`:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

2. **Use a production ASGI server**:
   ```bash
   gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Add authentication** if needed

4. **Set up HTTPS** with reverse proxy (nginx/Apache)

## Model Information

- **Architecture**: Actor-Critic neural networks
- **Training**: Reinforcement learning with Prioritized Experience Replay
- **Input**: 16-dimensional board encoding (log2 normalized)
- **Output**: Action probabilities for 4 directions

## Troubleshooting

### Models not found
Ensure models exist at: `models/trained_models_actor.pth` and `models/trained_models_critic.pth`

### Port already in use
Change the port in `api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Use 8080 instead
```

### Slow responses
The actor network is already optimized for speed. If you need even faster inference, ensure:
- Models are on the same device (CPU/GPU)
- No unnecessary logging
- Use batch processing for multiple requests

## Support

For issues or questions, please check:
- API docs at `/docs`
- Test with `python inference.py`
- Review server logs
