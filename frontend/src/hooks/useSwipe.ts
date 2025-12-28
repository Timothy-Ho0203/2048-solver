import { useRef, useEffect, RefObject } from 'react';
import type { Direction } from '../game/types';

interface UseSwipeOptions {
  onSwipe: (direction: Direction) => void;
  threshold?: number;
  enabled?: boolean;
}

interface TouchPosition {
  x: number;
  y: number;
}

/**
 * Custom hook for handling touch swipe gestures
 */
export function useSwipe(
  elementRef: RefObject<HTMLElement>,
  { onSwipe, threshold = 50, enabled = true }: UseSwipeOptions
) {
  const touchStart = useRef<TouchPosition | null>(null);
  const touchEnd = useRef<TouchPosition | null>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element || !enabled) return;

    const handleTouchStart = (e: TouchEvent) => {
      touchEnd.current = null;
      touchStart.current = {
        x: e.touches[0].clientX,
        y: e.touches[0].clientY,
      };
    };

    const handleTouchMove = (e: TouchEvent) => {
      touchEnd.current = {
        x: e.touches[0].clientX,
        y: e.touches[0].clientY,
      };
    };

    const handleTouchEnd = () => {
      if (!touchStart.current || !touchEnd.current) return;

      const deltaX = touchEnd.current.x - touchStart.current.x;
      const deltaY = touchEnd.current.y - touchStart.current.y;

      const absDeltaX = Math.abs(deltaX);
      const absDeltaY = Math.abs(deltaY);

      // Check if swipe meets threshold
      if (absDeltaX < threshold && absDeltaY < threshold) {
        return;
      }

      // Determine dominant direction
      if (absDeltaX > absDeltaY) {
        // Horizontal swipe
        onSwipe(deltaX > 0 ? 'right' : 'left');
      } else {
        // Vertical swipe
        onSwipe(deltaY > 0 ? 'down' : 'up');
      }

      // Reset
      touchStart.current = null;
      touchEnd.current = null;
    };

    element.addEventListener('touchstart', handleTouchStart, { passive: true });
    element.addEventListener('touchmove', handleTouchMove, { passive: true });
    element.addEventListener('touchend', handleTouchEnd);

    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchmove', handleTouchMove);
      element.removeEventListener('touchend', handleTouchEnd);
    };
  }, [elementRef, onSwipe, threshold, enabled]);
}
