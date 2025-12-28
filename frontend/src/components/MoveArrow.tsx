import type { Direction } from '../game/types';

interface MoveArrowProps {
  direction: Direction;
  confidence?: number;
}

export function MoveArrow({ direction, confidence }: MoveArrowProps) {
  // Calculate rotation based on direction
  const getRotation = (dir: Direction): string => {
    switch (dir) {
      case 'up':
        return 'rotate-0';
      case 'right':
        return 'rotate-90';
      case 'down':
        return 'rotate-180';
      case 'left':
        return 'rotate-[-90deg]';
    }
  };

  return (
    <div
      className="
        absolute inset-0
        flex items-center justify-center
        pointer-events-none
        z-10
      "
    >
      <div
        className={`
          transform ${getRotation(direction)}
          transition-all duration-300
        `}
      >
        <svg
          width="120"
          height="120"
          viewBox="0 0 120 120"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="drop-shadow-lg"
        >
          {/* Arrow body */}
          <rect
            x="50"
            y="30"
            width="20"
            height="60"
            rx="4"
            fill="rgba(59, 130, 246, 0.6)"
            stroke="rgba(59, 130, 246, 0.8)"
            strokeWidth="2"
          />
          {/* Arrow head */}
          <path
            d="M 60 10 L 90 40 L 70 40 L 70 50 L 50 50 L 50 40 L 30 40 Z"
            fill="rgba(59, 130, 246, 0.6)"
            stroke="rgba(59, 130, 246, 0.8)"
            strokeWidth="2"
          />
        </svg>
      </div>
      {confidence !== undefined && (
        <div
          className="
            absolute bottom-4
            bg-blue-500 bg-opacity-80
            text-white text-xs font-bold
            px-2 py-1 rounded
          "
        >
          {Math.round(confidence * 100)}%
        </div>
      )}
    </div>
  );
}
