import random
import numpy as np
from typing import List, Tuple, Sequence
from game import Direction


class Experience:
    """Container for a single experience tuple."""
    
    def __init__(self, state: List[List[int]], action: Direction, 
                 reward: float, next_state: List[List[int]], done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class SumTree:
    """Binary tree data structure where parent nodes are the sum of child priorities.

    Enables O(log N) update and O(log N) sampling by priority.
    """

    def __init__(self, capacity: int):
        # Next power of two for a complete binary tree implementation.
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity <<= 1

        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.data: List[Experience] = [None] * self.capacity  # type: ignore[assignment]
        self.write = 0
        self.n_entries = 0

    # ---------------------------------------------------------------------
    # Internal helper methods
    # ---------------------------------------------------------------------

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = idx >> 1
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find the highest tree index whose cumulative priority <= s."""
        left = idx << 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # ---------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------

    def total(self) -> float:  # noqa: D401 – single‑word summary is fine here
        """Return total priority mass."""
        return self.tree[1]

    def add(self, priority: float, data: Experience):
        """Add a new experience with `priority` into the tree."""
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update the priority of tree node `idx`."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Experience]:
        """Sample the tree so that each leaf is chosen proportional to its priority.

        Returns (tree index, priority, data).
        """
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer.

    Args:
        capacity: Maximum number of experiences to store.
        alpha: How much prioritization is used (0 -> uniform, 1 -> full PER).
        beta: Initial value for importance‑sampling (IS) correction.
        beta_increment_per_sampling: Amount to increment beta after each sample.
        epsilon: Small term to ensure all priorities are non‑zero.

    Reference:
        Schaul et al., "Prioritized Experience Replay", ICLR 2016.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 1e-4,
        epsilon: float = 1e-6,
    ):  # noqa: D401
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.max_priority = 1.0  # Ensures new experiences are always sampled at least once.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, experience: Experience, *, td_error: float | None = None):
        """Add a new experience.

        If `td_error` is not provided, the current maximum priority is used.
        """
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

        self.tree.add(priority, experience)

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], np.ndarray]:
        """Sample a batch of experiences along with their indices and IS weights."""
        if self.tree.n_entries == 0:
            raise ValueError("Cannot sample from an empty buffer – populate buffer first.")

        batch_size = min(batch_size, self.tree.n_entries)

        experiences: List[Experience] = []
        indices: List[int] = []
        priorities: List[float] = []

        total_p = self.tree.total()
        segment = total_p / batch_size
        eps = 1e-6  # small offset to keep s > 0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a + eps, b - eps) if segment > eps else random.uniform(eps, total_p)

            idx, priority, data = self.tree.get(s)

            # Fallback: If we (numerically) hit a zero‑priority leaf, resample once.
            if priority == 0.0:
                s = random.uniform(eps, total_p)
                idx, priority, data = self.tree.get(s)

            indices.append(idx)
            priorities.append(priority)
            experiences.append(data)

        # -------------------------- IS weights --------------------------
        sampling_probabilities = (np.array(priorities, dtype=np.float32) + self.epsilon) / (
            total_p + self.epsilon
        )
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        weights = (self.tree.n_entries * sampling_probabilities + self.epsilon) ** (-self.beta)
        weights /= weights.max()  # Normalize in [0,1]

        return experiences, indices, weights.astype(np.float32)

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]):
        """Update priorities of sampled transitions after learning."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:  # noqa: D401
        """Current number of stored experiences."""
        return self.tree.n_entries

    # ------------------------------------------------------------------
    # Optional convenience wrappers to mimic the original interface
    # ------------------------------------------------------------------

    def clear(self):
        """Remove all experiences and reset the buffer."""
        self.tree = SumTree(self.tree.capacity)
        self.max_priority = 1.0
