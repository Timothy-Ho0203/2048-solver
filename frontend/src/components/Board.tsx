import { useRef } from 'react';
import type { Board as BoardType, Direction } from '../game/types';
import { Tile } from './Tile';
import { MoveArrow } from './MoveArrow';

interface BoardProps {
  board: BoardType;
  suggestedMove?: Direction | null;
  confidence?: number;
}

export function Board({ board, suggestedMove, confidence }: BoardProps) {
  const boardRef = useRef<HTMLDivElement>(null);

  return (
    <div className="relative">
      <div
        ref={boardRef}
        className="
          bg-[#bbada0] rounded-lg p-3 md:p-4
          grid grid-cols-4 grid-rows-4 gap-2 md:gap-3
          touch-none
          w-[min(90vw,500px)] h-[min(90vw,500px)]
        "
        data-board
      >
        {board.map((row, rowIndex) =>
          row.map((value, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className="w-full h-full"
            >
              <Tile value={value} />
            </div>
          ))
        )}
      </div>
      {suggestedMove && (
        <MoveArrow direction={suggestedMove} confidence={confidence} />
      )}
    </div>
  );
}

export { type BoardType };
