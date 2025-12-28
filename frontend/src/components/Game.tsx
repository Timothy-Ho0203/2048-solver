import { useState, useCallback, useRef, useEffect } from 'react';
import { Board } from './Board';
import { ScoreBar } from './ScoreBar';
import { GameOverModal, WinModal } from './Modal';
import { useKeyboard } from '../hooks/useKeyboard';
import { useSwipe } from '../hooks/useSwipe';
import {
  initializeGame,
  move,
  spawnTile,
  canMove,
  has2048,
} from '../game/gameLogic';
import type { Direction } from '../game/types';
import { getOptimalMove } from '../api/moveApi';

const BEST_SCORE_KEY = 'best-score-2048';

export function Game() {
  const [board, setBoard] = useState(() => initializeGame());
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState(() => {
    const saved = localStorage.getItem(BEST_SCORE_KEY);
    return saved ? parseInt(saved, 10) : 0;
  });
  const [gameOver, setGameOver] = useState(false);
  const [won, setWon] = useState(false);
  const [showWinModal, setShowWinModal] = useState(false);
  const [suggestedMove, setSuggestedMove] = useState<Direction | null>(null);
  const [confidence, setConfidence] = useState<number | undefined>(undefined);

  const boardContainerRef = useRef<HTMLDivElement>(null);

  // Update best score in localStorage
  useEffect(() => {
    if (score > bestScore) {
      setBestScore(score);
      localStorage.setItem(BEST_SCORE_KEY, score.toString());
    }
  }, [score, bestScore]);

  // Fetch optimal move from API
  const fetchOptimalMove = useCallback(async (currentBoard: number[][]) => {
    try {
      const response = await getOptimalMove(currentBoard);
      if (response.best_move) {
        setSuggestedMove(response.best_move as Direction);
        setConfidence(response.confidence);
      } else {
        setSuggestedMove(null);
        setConfidence(undefined);
      }
    } catch (error) {
      console.error('Failed to fetch optimal move:', error);
      // Silently fail - don't show suggestion if API is unavailable
      setSuggestedMove(null);
      setConfidence(undefined);
    }
  }, []);

  // Fetch optimal move on initial load
  useEffect(() => {
    fetchOptimalMove(board);
  }, []);

  const handleMove = useCallback(
    (direction: Direction) => {
      if (gameOver || showWinModal) return;

      const result = move(board, direction);

      if (!result.moved) return;

      let newBoard = result.board;
      const newScore = score + result.scoreDelta;

      // Spawn new tile after successful move
      newBoard = spawnTile(newBoard);

      setBoard(newBoard);
      setScore(newScore);

      // Check win condition (only show modal once)
      if (!won && has2048(newBoard)) {
        setWon(true);
        setShowWinModal(true);
        return;
      }

      // Check game over condition
      if (!canMove(newBoard)) {
        setGameOver(true);
        setSuggestedMove(null);
      } else {
        // Fetch next optimal move after successful move
        fetchOptimalMove(newBoard);
      }
    },
    [board, score, gameOver, won, showWinModal, fetchOptimalMove]
  );

  const handleNewGame = useCallback(() => {
    const newBoard = initializeGame();
    setBoard(newBoard);
    setScore(0);
    setGameOver(false);
    setWon(false);
    setShowWinModal(false);
    fetchOptimalMove(newBoard);
  }, [fetchOptimalMove]);

  const handleContinueAfterWin = useCallback(() => {
    setShowWinModal(false);
  }, []);

  // Keyboard controls
  useKeyboard({
    onMove: handleMove,
    enabled: !gameOver && !showWinModal,
  });

  // Touch/swipe controls
  useSwipe(boardContainerRef, {
    onSwipe: handleMove,
    threshold: 50,
    enabled: !gameOver && !showWinModal,
  });

  return (
    <div className="game-container flex flex-col items-center w-full">
      <ScoreBar score={score} bestScore={bestScore} onNewGame={handleNewGame} />

      <div ref={boardContainerRef} className="w-full flex justify-center">
        <Board
          board={board}
          suggestedMove={suggestedMove}
          confidence={confidence}
        />
      </div>

      <GameOverModal
        isOpen={gameOver}
        onNewGame={handleNewGame}
        score={score}
      />

      <WinModal
        isOpen={showWinModal}
        onContinue={handleContinueAfterWin}
        onNewGame={handleNewGame}
        score={score}
      />
    </div>
  );
}
