import numpy as np

def position_occupied(position: np.ndarray, conformation: np.ndarray) -> bool:
    """
    Returns if position is present (or occupied) in the conformation NumPy array.

    Parameters:
    position: NumPy array of form [x,y]
    conformation: Numpy array of positions in above format
    """
    return any(np.array_equal(position, conformation_pos) for conformation_pos in conformation)

def initialize_positions(length: int) -> np.ndarray:
    """
    Generates a self-avoiding walk of a given length.

    Parameters:
    length: The length of the sequence for which the conformation is generated.

    Returns a NumPy array of 2D positions representing the conformation.
    """

    def self_avoiding_walk(conformation: np.ndarray) -> np.ndarray:
        """ Recursive helper function to generate a self-avoiding walk. """
        if len(conformation) == length:
            return conformation  # Return the conformation if the desired length is reached

        # Determine what moves are possible at current position, and randomly select one
        current_pos = conformation[-1]
        directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]  # Right, Up, Left, Down
        possible_moves = [np.add(current_pos, d) for d in directions if position_occupied(np.add(current_pos, d), conformation)==False]
        np.random.shuffle(possible_moves)

        # Add move to current conformation and continue
        for move in possible_moves:
            next_conformation = np.append(conformation, [move], axis=0)
            result = self_avoiding_walk(next_conformation)
            if result is not None:
                return result  # Return the successful path

        return None  # Or backtrack if no move is possible

    start_position = np.array([[0, 0]])
    return self_avoiding_walk(start_position)
