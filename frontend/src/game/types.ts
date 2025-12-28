export type Direction = 'up' | 'down' | 'left' | 'right';

export type Board = number[][];

export interface GameState {
  board: Board;
  score: number;
  gameOver: boolean;
  won: boolean;
  canMove: boolean;
}

export interface MoveResult {
  board: Board;
  scoreDelta: number;
  moved: boolean;
}

export interface Position {
  row: number;
  col: number;
}
