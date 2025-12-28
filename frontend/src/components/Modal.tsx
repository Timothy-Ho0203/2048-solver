import { ReactNode } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
}

export function Modal({ isOpen, onClose, children }: ModalProps) {
  if (!isOpen) return null;

  return (
    <div
      className="
        fixed inset-0 bg-black bg-opacity-50
        flex items-center justify-center
        z-50 p-4
      "
      onClick={onClose}
    >
      <div
        className="
          bg-white rounded-2xl p-6 md:p-8
          max-w-md w-full
          shadow-2xl
          transform transition-all
        "
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}

interface GameOverModalProps {
  isOpen: boolean;
  onNewGame: () => void;
  score: number;
}

export function GameOverModal({ isOpen, onNewGame, score }: GameOverModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onNewGame}>
      <div className="text-center">
        <h2 className="text-4xl font-bold text-[#776e65] mb-4">Game Over!</h2>
        <p className="text-xl text-[#776e65] mb-2">
          Your score: <strong>{score}</strong>
        </p>
        <p className="text-[#776e65] mb-6">
          No more moves available. Try again!
        </p>
        <button
          onClick={onNewGame}
          className="
            bg-[#8f7a66] hover:bg-[#9f8a76] text-white
            font-bold py-3 px-8 rounded-lg
            transition-colors duration-200
            w-full
          "
        >
          New Game
        </button>
      </div>
    </Modal>
  );
}

interface WinModalProps {
  isOpen: boolean;
  onContinue: () => void;
  onNewGame: () => void;
  score: number;
}

export function WinModal({
  isOpen,
  onContinue,
  onNewGame,
  score,
}: WinModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onContinue}>
      <div className="text-center">
        <h2 className="text-4xl font-bold text-[#f59563] mb-4">
          You Win! 🎉
        </h2>
        <p className="text-xl text-[#776e65] mb-2">
          Your score: <strong>{score}</strong>
        </p>
        <p className="text-[#776e65] mb-6">
          You reached 2048! Keep playing to get a higher score.
        </p>
        <div className="flex flex-col gap-3">
          <button
            onClick={onContinue}
            className="
              bg-[#8f7a66] hover:bg-[#9f8a76] text-white
              font-bold py-3 px-8 rounded-lg
              transition-colors duration-200
              w-full
            "
          >
            Keep Playing
          </button>
          <button
            onClick={onNewGame}
            className="
              bg-[#eee4da] hover:bg-[#ede0c8] text-[#776e65]
              font-bold py-3 px-8 rounded-lg
              transition-colors duration-200
              w-full
            "
          >
            New Game
          </button>
        </div>
      </div>
    </Modal>
  );
}
