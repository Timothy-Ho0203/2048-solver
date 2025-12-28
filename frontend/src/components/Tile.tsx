interface TileProps {
  value: number;
}

const getTileColor = (value: number): string => {
  const colors: { [key: number]: string } = {
    2: 'bg-[#eee4da] text-gray-700',
    4: 'bg-[#ede0c8] text-gray-700',
    8: 'bg-[#f2b179] text-white',
    16: 'bg-[#f59563] text-white',
    32: 'bg-[#f67c5f] text-white',
    64: 'bg-[#f65e3b] text-white',
    128: 'bg-[#edcf72] text-white',
    256: 'bg-[#edcc61] text-white',
    512: 'bg-[#edc850] text-white',
    1024: 'bg-[#edc53f] text-white',
    2048: 'bg-[#edc22e] text-white',
    4096: 'bg-[#3c3a32] text-white',
    8192: 'bg-[#3c3a32] text-white',
  };

  return colors[value] || 'bg-[#3c3a32] text-white';
};

const getFontSize = (value: number): string => {
  if (value >= 1024) return 'text-2xl md:text-3xl';
  if (value >= 128) return 'text-3xl md:text-4xl';
  return 'text-4xl md:text-5xl';
};

export function Tile({ value }: TileProps) {
  if (value === 0) {
    return (
      <div className="w-full h-full bg-[#cdc1b4] bg-opacity-50 rounded-lg"></div>
    );
  }

  return (
    <div
      className={`
        w-full h-full rounded-lg flex items-center justify-center
        font-bold transition-all duration-200
        ${getTileColor(value)}
        ${getFontSize(value)}
        animate-pop
      `}
    >
      {value}
    </div>
  );
}
