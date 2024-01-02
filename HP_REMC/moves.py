import numpy as np

def position_occupied(position: np.ndarray, conformation: np.ndarray) -> bool:
    """
    Returns if position is in positions NumPy array.

    Parameters:
    position: NumPy array of form [x,y]
    conformation: Numpy array of positions in above format
    
    Returns a boolean indicating if position is occupied (True) or not (False)
    """
    return any(np.array_equal(position, conformation_pos) for conformation_pos in conformation)

def end_move(positions: np.ndarray, index: int) -> np.ndarray:
    """
    Moves the end position of a structure consistent with the VSHD neighborhood.

    Parameters:
    positions: NumPy array of positions.  This array is modified in-place
    index: Index of the position to be moved.

    Returns the updated positions NumPy array, or original if end move is not possible.
    """

    # Check if index is at the ends
    if index in {0, len(positions) - 1}:
        # Determine possible adjacent positions
        pivot_index = 1 if index == 0 else len(positions) - 2
        current_pos = positions[pivot_index]

        # Same logic as initializing positions
        directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]  # Right, Up, Left, Down
        possible_moves = [np.add(current_pos, d) for d in directions if not position_occupied(np.add(current_pos, d), positions)]
        np.random.shuffle(possible_moves)

        # Perform end-move if unoccupied adjacent space is found; should be random due to shuffle
        if possible_moves:
            positions[index] = possible_moves[0]

    return positions

def crankshaft(positions: np.ndarray, index: int) -> np.ndarray:
    """
    Performs a crankshaft movement on a structure if possible.

    Parameters:
    positions: Array of positions. This array is modified in-place
    index: Index at which the crankshaft movement is attempted.

    Returns the updated positions NumPy array, or original if crankshaft move is not possible.
    """

    # Ensure index is valid for crankshaft movement
    if index == 0 or index >= len(positions) - 2:
        return positions

    # Extract relevant positions (i-1, i, i+1, i+2) and check if move is possible
    im1, i, ip1, ip2 = positions[index - 1], positions[index], positions[index + 1], positions[index + 2]
    if np.linalg.norm(im1 - ip2) != 1:
        return positions

    # Calculate new positions, check if occupied, and update positions
    i_prime, ip1_prime = im1 + (im1 - i), ip2 + (ip2 - ip1)
    if position_occupied(i_prime, positions) or position_occupied(ip1_prime, positions):
        return positions
    positions[index], positions[index + 1] = i_prime, ip1_prime
    return positions

def corner_move(positions: np.ndarray, index: int) -> np.ndarray:
    """
    Performs a corner move on the given positions array, consistent with VSHD neighborhood.

    Parameters:
    positions: Numpy array of positions. This array is modified in-place
    index: Index of position in corner.

    Returns the updated NumPy array of positions, or original if end move is not possible
    """
    # Check if index is valid for corner move
    if index == {0, len(positions) - 1}:
        return positions
    elif np.linalg.norm(positions[index + 1] - positions[index - 1]) != np.sqrt(2):
        return positions

    # Get relevant positions (i, i-1, i+1)
    i, im1, ip1 = positions[index], positions[index - 1], positions[index + 1]

    # Compute the new position, check if occupied, and update
    new_pos = np.array([im1[dim] if ip1[dim] == i[dim] else ip1[dim] for dim in range(2)])
    if not position_occupied(new_pos, positions):
        positions[index] = new_pos

    return positions


def crankshaft(positions: np.ndarray, index: int) -> np.ndarray:
    """
    Performs a crankshaft movement on a structure if possible, consistent with VSHD neighborhood.

    Parameters:
    positions: Array of positions. This array is modified in-place
    index: Index at which the crankshaft movement is attempted.

    Returns the updated positions NumPy array, or original if crankshaft move is not possible
    """

    # Ensure index is valid for crankshaft movement
    if index == 0 or index >= len(positions) - 2:
        return positions

    # Extract relevant positions (i-1, i, i+1, i+2) and check if move is possible
    im1, i, ip1, ip2 = positions[index - 1], positions[index], positions[index + 1], positions[index + 2]
    if np.linalg.norm(im1 - ip2) != 1:
        return positions

    # Calculate new positions and check if occupied; if not, update positions
    i_prime, ip1_prime = im1 + (im1 - i), ip2 + (ip2 - ip1)
    if position_occupied(i_prime, positions) or position_occupied(ip1_prime, positions):
        return positions
    positions[index], positions[index + 1] = i_prime, ip1_prime
    return positions

def pull_move(positions: np.ndarray, index: int) -> np.ndarray:
    """
    Performs a pull move on the given positions array.

    Parameters:
    positions: Numpy array of positions. This array is modified in-place
    index: Index of the position to perform the pull move.

    Returns the updated positions NumPy array, or original if pull move is not possible
    """
    # Check if corner move is possible (at least 2 from either end)
    if index < 2 or index > len(positions) - 3:
        return positions

    # Check if a corner move is possible and perform it if so (3a)
    if index not in {0, len(positions) - 1} and np.linalg.norm(positions[index + 1] - positions[index - 1]) == np.sqrt(2):
        corner_move(positions, index)
        return positions

    # Define relevant positions (i-1, i-1, i, i+1)
    im2, im1, i, ip1, ip2 = positions[index - 2], positions[index - 1], positions[index], positions[index + 1], positions[index + 2]

    # Calculate possible positions of C
    i_ip1_vec = ip1 - i
    i_C_vec = i_ip1_vec[::-1]
    C_positions = [i + i_C_vec, i - i_C_vec]
    C_positions = sorted(C_positions, key=lambda x: np.linalg.norm(x - im2)) # sort by distance to i-2 (for scenario in 3b)

    # Randomly choose to first attempt forward or reverse move
    move_attempts = ['forward', 'reverse']
    np.random.shuffle(move_attempts)

    for attempt in move_attempts:
        if attempt == 'forward':
            # Attempt forward
            for C in C_positions:
                # Calculate corresponding L for this C
                L = np.array([ip1[0] if C[0] == i[0] else C[0], ip1[1] if C[1] == i[1] else C[1]])

                if position_occupied(C, positions) or position_occupied(L, positions):
                    continue

                # Simple pull move (3b); if possible, sorting ensures this occurs for first C
                if np.linalg.norm(im2 - C) == 1.0:
                    positions[index], positions[index - 1] = L, C

                # Complex pull move (3c)
                else:
                    original_positions = positions.copy()
                    positions[index], positions[index - 1] = L, C
                    i = index - 2
                    while i >= 0 and (np.linalg.norm(original_positions[i]-positions[i + 1])!= 1):
                        positions[i] = original_positions[i + 2]
                        i -= 1
                return positions

        else:
            # Attempt reverse
            for C in C_positions:
                # Calculate corresponding L for this C in the reverse direction
                L = np.array([im1[0] if C[0] == i[0] else C[0], im1[1] if C[1] == i[1] else C[1]])

                if position_occupied(C, positions) or position_occupied(L, positions):
                    continue

                # Check for reverse pull move conditions
                if index < len(positions) - 3 and np.linalg.norm(ip2 - C) == 1.0:
                    # Simple pull move in reverse direction
                    positions[index + 1], positions[index] = C, L
                else:
                    # Complex pull move in reverse direction
                    original_positions = positions.copy()
                    positions[index + 1], positions[index] = C, L
                    i = index + 2
                    while i <= len(positions) - 1 and np.linalg.norm(original_positions[i] - positions[i - 1]) != 1:
                        positions[i] = original_positions[i - 2]
                        i += 1
                return positions

    # return original if move is not possible
    return positions