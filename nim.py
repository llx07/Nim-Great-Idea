from abc import ABC, abstractmethod

class NimEnv:
    """The environment for Nim.

    Attributes:
        piles: a list of int for the number of stones in each pile
        player: the current player (`0` or `1`)
        winner: the winner of the game, or `None` if the game is till in process
    """

    WIN_REWARD = 100.0
    ELSE_REWARD = 0.0

    def __init__(self, initial: None | list[int] = None) -> None:
        """Initialize the environment.

        Args:
            initial: the initial setting of the piles, default set to `[1, 3, 5, 7]`.
        """
        if initial is None:
            self.piles = [1, 3, 5, 7]
        else:
            self.piles = initial
        self.player = 0
        self.winner = None
    
    def get_available_actions(self) -> list[tuple[int, int]]:
        """Return all available actions according to `self.piles`.

        Returns:
            A list of tuple in format (pile_number, count).
        """
        actions = []
        for pile_number, stones in enumerate(self.piles):
            for count in range(1, stones + 1):
                actions.append((pile_number,count))
        return actions

    def move(self, action: tuple[int, int]) -> float:
        """Make the move `action`.

        This method will update `self.piles`, `self.player` and `self.winner`.

        If `self.winner` is not None, calling this method will cause an
        RuntimeError. If `action` is invalid, calling the method will cause
        an ValueError.
        """
        if self.winner is not None:
            raise RuntimeError("The game has ended")
        
        pile_number, count = action
        if pile_number < 0 or pile_number >= 4:
            raise ValueError(f"Invalid pile_num:{pile_number}")
        if count < 1 or count > self.piles[pile_number]:
            raise ValueError(f"Invalid stone count:{count} in pile:{pile_number} which has {self.piles[pile_number]} stones")
        """
        Args:
            action: the action in format (pile_number, count)
        """
        self.piles[pile_number] -= count
        
        if all(stones == 0 for stones in self.piles):
            self.winner = self.player
            reward = self.WIN_REWARD

        else:
            self.winner = 1 - self.player
            reward = self.ELSE_REWARD
        
        return reward
        """
        Returns:
            A float value for reward. return `WIN_REWARD` if the current player
            wins, return `ELSE_REWARD` otherwise.

        Raises:
            ValueError: if the action is invalid.
            RuntimeError: if the game is already end.
        """

class Agent(ABC):
    """
    Abstract Base Class for AI
    """
    @abstractmethod
    def choose_action(self, state: NimEnv) -> tuple[int, int]:
        pass
