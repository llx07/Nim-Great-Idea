import nim
import pickle


class QLearningAgent(nim.Agent):
    def __init__(self) -> None:
        """Initialize the agent. Load q-table from file (if exist)
        Also set alpha and epsilon.
        """
        self.q = dict()

    def save(self) -> None:
        """Save the q-table into file."""
        # See the module pickle
        raise NotImplementedError

    def update(
        self,
        state_before: list[int],
        action: tuple[int, int],
        state_after: list[int],
        reward: float,
    ):
        """Update the Q value for the state
        """
        raise NotImplementedError
    
    def choose_action(self, state: nim.NimEnv) -> tuple[int, int]:
        """Given a state `state`, return an action.
        """
        raise NotImplementedError

if __name__ == "__main__":
    # Train the agent here.
    raise NotImplementedError