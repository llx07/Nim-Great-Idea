from abc import ABC, abstractmethod

class NimEnv:
    def __init__(self, initial: None | list[int] = None) -> None:
        """
        Initialize the environment.
        """
        if initial is None:
            self.piles = [1, 3, 5, 7]
        else:
            self.piles = initial
        self.player = 0
        self.winner = None
    
    def available_actions(self) -> list[tuple[int, int]]:
        """
        Return all available actions according to `self.piles`.
        """
        raise NotImplementedError()

    def move(self, action: tuple[int, int]):
        """
        Make the move `action`.
        """
        raise NotImplementedError()
    


"""
Abstract Base Class for AI
"""
class Agent(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def choose_action(self, state) -> tuple[int, int]:
        pass
