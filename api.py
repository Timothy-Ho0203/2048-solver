#!/usr/bin/env python3
"""
FastAPI endpoint for 2048 move inference.
Provides a REST API to get optimal moves from trained models.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import uvicorn

from inference import GameInferenceEngine
from game import Game2048


# Initialize FastAPI app
app = FastAPI(
    title="2048 AI Move API",
    description="Get optimal moves for 2048 game using trained Actor-Critic models",
    version="1.0.0"
)

# Add CORS middleware to allow requests from web browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (loaded once at startup)
engine: Optional[GameInferenceEngine] = None


# Request/Response models
class BoardState(BaseModel):
    """Board state for move query."""
    board: List[List[int]] = Field(
        ...,
        description="4x4 board state with tile values (0 for empty)",
        example=[[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0]]
    )

    @validator('board')
    def validate_board(cls, v):
        """Validate board dimensions and values."""
        if len(v) != 4:
            raise ValueError("Board must have exactly 4 rows")
        for row in v:
            if len(row) != 4:
                raise ValueError("Each row must have exactly 4 columns")
            for cell in row:
                if not isinstance(cell, int) or cell < 0:
                    raise ValueError("All cell values must be non-negative integers")
        return v


class MoveResponse(BaseModel):
    """Response containing the optimal move."""
    best_move: Optional[str] = Field(
        ...,
        description="Best move direction (up/down/left/right) or null if no valid moves"
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the best move (0.0 to 1.0)"
    )
    all_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for all valid moves"
    )


class GameInfo(BaseModel):
    """Additional game information."""
    score: int
    max_tile: int
    valid_moves: List[str]


class DetailedMoveResponse(MoveResponse):
    """Extended response with game information."""
    game_info: GameInfo


# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load models when the API starts."""
    global engine
    print("Loading AI models...")
    try:
        engine = GameInferenceEngine()
        print("AI models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "running",
        "message": "2048 AI Move API is running",
        "models_loaded": engine is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy"}


# Main inference endpoint
@app.post("/get_move", response_model=MoveResponse)
async def get_move(board_state: BoardState):
    """
    Get the optimal next move for a given board state.

    Args:
        board_state: Current 4x4 board configuration

    Returns:
        MoveResponse: Best move, confidence, and all probabilities

    Example request:
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

    Example response:
    ```json
    {
        "best_move": "left",
        "confidence": 0.85,
        "all_probabilities": {
            "left": 0.85,
            "up": 0.10,
            "right": 0.03,
            "down": 0.02
        }
    }
    ```
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        result = engine.get_move_from_board(board_state.board)
        return MoveResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing board: {str(e)}")


@app.post("/get_move_detailed", response_model=DetailedMoveResponse)
async def get_move_detailed(board_state: BoardState):
    """
    Get the optimal move with additional game information.

    Args:
        board_state: Current 4x4 board configuration

    Returns:
        DetailedMoveResponse: Move information plus game state details
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Create game instance to get additional info
        game = Game2048()
        game.board = [row[:] for row in board_state.board]

        # Get move prediction
        result = engine.get_move_from_board(board_state.board)

        # Get game info
        game_info = GameInfo(
            score=game.get_score(),
            max_tile=game.get_max_tile(),
            valid_moves=[move.value for move in game.get_valid_moves()]
        )

        return DetailedMoveResponse(
            **result,
            game_info=game_info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing board: {str(e)}")


# Batch processing endpoint
@app.post("/get_moves_batch")
async def get_moves_batch(boards: List[BoardState]):
    """
    Get optimal moves for multiple board states in a single request.

    Args:
        boards: List of board states

    Returns:
        List of move responses
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    results = []
    for board_state in boards:
        try:
            result = engine.get_move_from_board(board_state.board)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "best_move": None,
                "confidence": 0.0,
                "all_probabilities": {}
            })

    return results


if __name__ == "__main__":
    print("Starting 2048 AI Move API server...")
    print("API documentation available at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("\nExample curl command:")
    print('curl -X POST "http://localhost:8000/get_move" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"board": [[2,0,0,0],[0,0,0,0],[0,0,0,0],[4,0,0,0]]}\'')
    print("\n" + "="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
