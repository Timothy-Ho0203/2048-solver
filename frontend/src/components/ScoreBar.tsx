interface ScoreBarProps {
  score: number;
  bestScore: number;
  onNewGame: () => void;
}

export function ScoreBar({ score, bestScore, onNewGame }: ScoreBarProps) {
  return (
    <div className="w-full max-w-[500px] mb-4">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-6xl font-bold text-[#776e65]">2048</h1>
        <div className="flex gap-2">
          <div className="bg-[#bbada0] rounded-lg px-4 py-2 text-center min-w-[80px]">
            <div className="text-xs text-[#eee4da] uppercase font-bold">
              Score
            </div>
            <div className="text-2xl font-bold text-white">{score}</div>
          </div>
          <div className="bg-[#bbada0] rounded-lg px-4 py-2 text-center min-w-[80px]">
            <div className="text-xs text-[#eee4da] uppercase font-bold">
              Best
            </div>
            <div className="text-2xl font-bold text-white">{bestScore}</div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-[#776e65] text-sm md:text-base">
          Join the numbers and get to the <strong>2048 tile!</strong>
        </p>
        <button
          onClick={onNewGame}
          className="
            bg-[#8f7a66] hover:bg-[#9f8a76] text-white
            font-bold py-2 px-4 rounded-lg
            transition-colors duration-200
            text-sm md:text-base
            whitespace-nowrap
          "
        >
          New Game
        </button>
      </div>

      <div className="mt-4 text-[#776e65] text-sm">
        <p>
          <strong>HOW TO PLAY:</strong> Use your <strong>arrow keys</strong> or{' '}
          <strong>WASD</strong> to move the tiles. On mobile,{' '}
          <strong>swipe</strong> in any direction. Tiles with the same number
          merge into one when they touch. Add them up to reach{' '}
          <strong>2048</strong>!
        </p>
      </div>
    </div>
  );
}
