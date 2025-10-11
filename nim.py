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
        raise NotImplementedError() # TODO

    def move(self, action: tuple[int, int]):
        """Make the move `action`.

        This method will update `self.piles`, `self.player` and `self.winner`.

        If `self.winner` is not None, calling this method will cause an
        RuntimeError. If `action` is invalid, calling the method will cause
        an ValueError.

        Args:
            action: the action in format (pile_number, count)

        Raises:
            ValueError: if the action is invalid.
            RuntimeError: if the game is already end.
        """
        raise NotImplementedError() # TODO

    

class Agent(ABC):
    """
    Abstract Base Class for AI
    """
    @abstractmethod
    def choose_action(self, state: NimEnv) -> tuple[int, int]:
        pass
