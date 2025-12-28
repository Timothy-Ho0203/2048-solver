import { useEffect } from 'react';
import type { Direction } from '../game/types';

interface UseKeyboardOptions {
  onMove: (direction: Direction) => void;
  enabled?: boolean;
}

/**
 * Custom hook for handling keyboard input (Arrow keys + WASD)
 */
export function useKeyboard({ onMove, enabled = true }: UseKeyboardOptions) {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      let direction: Direction | null = null;

      switch (event.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
          direction = 'up';
          break;
        case 'ArrowDown':
        case 's':
        case 'S':
          direction = 'down';
          break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
          direction = 'left';
          break;
        case 'ArrowRight':
        case 'd':
        case 'D':
          direction = 'right';
          break;
      }

      if (direction) {
        event.preventDefault(); // Prevent default scrolling behavior
        onMove(direction);
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [onMove, enabled]);
}
