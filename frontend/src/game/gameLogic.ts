import type { Board, Direction, MoveResult } from './types';

/**
 * Creates an empty 4x4 board
 */
export function createEmptyBoard(): Board {
  return Array(4)
    .fill(null)
    .map(() => Array(4).fill(0));
}

/**
 * Creates a deep copy of the board
 */
export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row]);
}

/**
 * Gets all empty cells on the board
 */
function getEmptyCells(board: Board): { row: number; col: number }[] {
  const empty: { row: number; col: number }[] = [];
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      if (board[row][col] === 0) {
        empty.push({ row, col });
      }
    }
  }
  return empty;
}

/**
 * Spawns a new tile (90% chance of 2, 10% chance of 4) on an empty cell
 */
export function spawnTile(board: Board): Board {
  const newBoard = cloneBoard(board);
  const emptyCells = getEmptyCells(newBoard);

  if (emptyCells.length === 0) {
    return newBoard;
  }

  const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
  const value = Math.random() < 0.9 ? 2 : 4;

  newBoard[randomCell.row][randomCell.col] = value;
  return newBoard;
}

/**
 * Rotates the board 90 degrees clockwise
 */
function rotateBoard(board: Board): Board {
  const n = board.length;
  const rotated = createEmptyBoard();

  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      rotated[col][n - 1 - row] = board[row][col];
    }
  }

  return rotated;
}

/**
 * Compresses non-zero values to the left, maintaining order
 */
function compressLeft(row: number[]): number[] {
  const filtered = row.filter((val) => val !== 0);
  const zeros = Array(row.length - filtered.length).fill(0);
  return [...filtered, ...zeros];
}

/**
 * Merges adjacent equal values from left to right (only once per tile per move)
 * Returns the merged row and the score gained
 */
function mergeLeft(row: number[]): { row: number[]; score: number } {
  const result = [...row];
  let score = 0;

  for (let i = 0; i < result.length - 1; i++) {
    if (result[i] !== 0 && result[i] === result[i + 1]) {
      result[i] = result[i] * 2;
      result[i + 1] = 0;
      score += result[i];
      i++; // Skip next cell to prevent double merge
    }
  }

  return { row: result, score };
}

/**
 * Processes a single row: compress -> merge -> compress
 */
function processRowLeft(row: number[]): { row: number[]; score: number } {
  let compressed = compressLeft(row);
  const { row: merged, score } = mergeLeft(compressed);
  compressed = compressLeft(merged);

  return { row: compressed, score };
}

/**
 * Checks if two boards are equal
 */
function boardsEqual(board1: Board, board2: Board): boolean {
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      if (board1[row][col] !== board2[row][col]) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Performs a move in the specified direction
 */
export function move(board: Board, direction: Direction): MoveResult {
  let workingBoard = cloneBoard(board);
  let rotations = 0;

  // Rotate board so we always process as "left" move
  switch (direction) {
    case 'left':
      rotations = 0;
      break;
    case 'up':
      rotations = 3;
      workingBoard = rotateBoard(
        rotateBoard(rotateBoard(workingBoard))
      );
      break;
    case 'right':
      rotations = 2;
      workingBoard = rotateBoard(rotateBoard(workingBoard));
      break;
    case 'down':
      rotations = 1;
      workingBoard = rotateBoard(workingBoard);
      break;
  }

  // Process each row
  let totalScore = 0;
  const processedBoard = workingBoard.map((row) => {
    const { row: newRow, score } = processRowLeft(row);
    totalScore += score;
    return newRow;
  });

  // Rotate back to original orientation
  let finalBoard = processedBoard;
  for (let i = 0; i < (4 - rotations) % 4; i++) {
    finalBoard = rotateBoard(finalBoard);
  }

  const moved = !boardsEqual(board, finalBoard);

  return {
    board: finalBoard,
    scoreDelta: totalScore,
    moved,
  };
}

/**
 * Checks if any moves are possible
 */
export function canMove(board: Board): boolean {
  // Check if there are any empty cells
  if (getEmptyCells(board).length > 0) {
    return true;
  }

  // Check if any adjacent cells can merge
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const current = board[row][col];

      // Check right
      if (col < 3 && current === board[row][col + 1]) {
        return true;
      }

      // Check down
      if (row < 3 && current === board[row + 1][col]) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Checks if the board has a 2048 tile
 */
export function has2048(board: Board): boolean {
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      if (board[row][col] === 2048) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Gets all valid moves from the current board state
 */
export function getValidMoves(board: Board): Direction[] {
  const directions: Direction[] = ['up', 'down', 'left', 'right'];
  return directions.filter((dir) => {
    const result = move(board, dir);
    return result.moved;
  });
}

/**
 * Initializes a new game board with 2 random tiles
 */
export function initializeGame(): Board {
  let board = createEmptyBoard();
  board = spawnTile(board);
  board = spawnTile(board);
  return board;
}
