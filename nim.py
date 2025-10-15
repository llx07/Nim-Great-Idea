from abc import ABC, abstractmethod


class NimEnv:
    """The environment for Nim.

    Attributes:
        piles: a list of int for the number of stones in each pile
        player: the current player (`0` or `1`)
        winner: the winner of the game, or `None` if the game is till in process
    """

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
                actions.append((pile_number, count))
        return actions

    def move(self, action: tuple[int, int]) -> int:
        """Make the move `action`.

        If the action is invalid, this method will return -1 and do nothing.
        Otherwise, this method will update `self.piles`, `self.player` and 
        `self.winner`.


        Args:
            action: the action in format (pile_number, count)

        Returns:
            A int indicate the game status.
            `-1` for invalid action, `1` for game continue, `0` for game ends.
        """
        if self.winner is not None:
            return -1

        pile_number, count = action
        if pile_number < 0 or pile_number >= len(self.piles):
            return -1

        if count < 1 or count > self.piles[pile_number]:
            return -1

        self.piles[pile_number] -= count

        if all(stones == 0 for stones in self.piles):
            self.winner = self.player
            return 0

        else:
            self.player ^= 1
            return 1


class Agent(ABC):
    """
    Abstract Base Class for AI
    """

    @abstractmethod
    def choose_action(self, state: NimEnv) -> tuple[int, int]:
        pass
