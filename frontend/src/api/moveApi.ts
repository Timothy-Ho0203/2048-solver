import type { Board } from '../game/types';

const API_BASE_URL = 'http://localhost:8000';

export interface MoveResponse {
  best_move: string | null;
  confidence: number;
  all_probabilities: Record<string, number>;
}

/**
 * Fetches the optimal move from the backend API
 */
export async function getOptimalMove(board: Board): Promise<MoveResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/get_move`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ board }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data: MoveResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching optimal move:', error);
    throw error;
  }
}

/**
 * Checks if the API is available
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
}
