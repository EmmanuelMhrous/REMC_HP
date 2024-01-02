import numpy as np
from typing import Optional
from .moves import end_move, corner_move, crankshaft, pull_move
from .initialize import initialize_positions

def energy(positions: np.ndarray, sequence: str) -> int:
    """
    Calculate the energy based on positions and the sequence.

    Parameters:
    positions: Numpy array of positions; this is not modified
    sequence: String consisting only of "H" or "P" characters to indicate Hydrophobic or Polar

    Returns the calculated energy value.
    """
    energy = 0
    for i in range(len(positions) - 2):
        for j in range(i + 2, len(positions)):
            if sequence[i] == "H" and sequence[j] == "H" and np.linalg.norm(positions[i] - positions[j]) == 1:
                energy -= 1
    return energy

def make_move(positions: np.ndarray, index: int, pull_prob: float) -> np.ndarray:
    """
    Determines and performs a move on the positions array based on the index.

    Parameters:
    positions: Numpy array of positions; this is modified in-place
    index: Index for the move.
    pull_prob: Probability of performing a move in the pull neighborhood;
               there is a (1 - pull_prob) probability of a move in VSHD neighborhood

    Returns the updated positions array after the move.
    """
    # Perform move in VSHD neighborhood with 1-pull_prob probability
    if np.random.uniform() > pull_prob:
        # Perform end move if index is at end
        if index in {0, len(positions) - 1}:
            return end_move(positions, index)

        # Define relevant positions otherwise: i-1, i, i+1
        im1, i, ip1 = positions[index - 1], positions[index], positions[index + 1]

        # If in a corner, determine if crankshaft is possible, or perform simple corner_move otherwise
        if np.linalg.norm(im1 + ip1) == np.sqrt(2):
            if index < len(positions) - 2 and np.linalg.norm(positions[index + 2] - im1) == 1:
                return crankshaft(positions, index)
            return corner_move(positions, index)

        return positions

    return pull_move(positions, index)

def MCSearch(num_steps: int, conformation: np.ndarray, sequence: str,
             conformation_temp: float, pull_prob: float, 
             E_star: Optional[float] = None) -> np.ndarray:
    """
    Monte Carlo search to find the conformation with the lowest energy.

    Parameters:
    num_steps: Number of steps in the Monte Carlo search.
    conformation: Initial positions array. This is NOT modified in-place
    sequence: String consisting only of "H" or "P" characters to indicate Hydrophobic or Polar
    conformation_temp: Temperature parameter for the Monte Carlo algorithm.
    pull_prob: Probability of performing a move in the pull neighborhood;
               there is a (1 - pull_prob) probability of a move in VSHD neighborhood
    E_star (optional): Optimal energy for early stopping, if known; usually for benchmark sequences

    Returns the last conformation NumPy array which attained the lowest energy of the search
    """
    # Boltzmann constant
    k_b = 0.001987

    # Initialize best (lowest) energy conformation
    best_conformation = conformation.copy()
    best_energy = energy(best_conformation, sequence)

    for step in range(num_steps):
        # A move is made on a random index of the conformation
        # A copy is used to preserve original if the Metropolis criterion is not met
        c_prime = conformation.copy()
        i = np.random.randint(len(c_prime))
        c_prime = make_move(c_prime, i, pull_prob)
        E = energy(conformation, sequence)
        E_prime = energy(c_prime, sequence)

        # Update conformation if conditions are met
        if E_prime <= E or np.random.uniform() < np.exp(-(E_prime - E) / (k_b * conformation_temp)):
            conformation = c_prime

            # Update best conformation and energy if a new best is found
            if E_prime <= best_energy:
                best_conformation = c_prime
                best_energy = E_prime

            # Process stops if minimum energy is known and is attained 
            # (otherwise it may go to higher energy due to Metropolis Criterion)
            if E_star is not None and E_prime <= E_star:
                break

    return best_conformation


def REMC(sequence: str, max_steps: int, num_MC_steps: int, min_temp: float, 
         max_temp: float, num_replicas: int, pull_prob: float, 
         update_freq: Optional[int] = 10, E_star: Optional[float] = None) -> np.ndarray:
    """
    Replica Exchange Monte Carlo algorithm.

    Parameters:
    sequence: String consisting only of "H" or "P" characters to indicate Hydrophobic or Polar.
    max_steps: Maximum number of steps for the algorithm.
    num_MC_steps: Number of Monte Carlo steps per replica.
    min_temp: Minimum temperature (in Kelvin).
    max_temp: Maximum temperature (in Kelvin).
    num_replicas: Number of replicas.
    pull_prob: Probability of performing a move in the pull neighborhood;
               there is a (1 - pull_prob) probability of a move in VSHD neighborhood
    update_freq: Number of steps that energies array is outputted. Default is every 10 steps.
    E_star (optional): Optimal energy for early stopping, if known; usually for benchmark sequences
    
    Returns: the Numpy array of the conformation with the lowest energy
    """
    # boltzmann constant (kcal/(mol·K))
    k_b = 0.001987

    # Initialize arrays of temperatures, conformations, and energies
    temperatures = np.linspace(min_temp, max_temp, num_replicas)
    conformations = [initialize_positions(len(sequence)) for _ in range(num_replicas)]
    offset = 0
    energies = [energy(conformation, sequence) for conformation in conformations]
    energies_history = [energies]
    best_energy = 0
    best_conformation = None
    # energies_history = [energies] #used for logging
    
    for step in range(max_steps):
        # Perform Monte Carlo Search for each replica
        for index, conformation in enumerate(conformations):
            new_conformation = MCSearch(num_MC_steps, conformation, sequence, temperatures[index], pull_prob, E_star)
            new_energy = energy(new_conformation, sequence)
            if new_energy <= energies[index]:
                conformations[index] = new_conformation
                energies[index] = new_energy

        # Perform temperature swapping between replicas
        for i in range(offset, num_replicas - 1, 2):
            j = i + 1
            delta = (1 / (k_b * temperatures[j]) - (1 / (k_b * temperatures[i]))) * (energies[j] - energies[i])
            if delta <= 0 or np.random.uniform() <= np.exp(-delta):
                temperatures[i], temperatures[j] = temperatures[j], temperatures[i]

        offset = 1 - offset  # Alternate offset

        # Print status, and check if minimum energy was met
        if step % update_freq == 0:
            print(f"Current step: {step}, Energies: {energies}")
        # energies_history.append(energies.copy()) # used for logging
        E_min = min(energies)
        if E_star is not None and E_min <= E_star:
            print("Minimum energy achieved!")
            best_energy = E_min
            best_conformation = conformations[np.argmin(energies)]
            break
        if E_min <= best_energy:
            best_energy = E_min
            best_conformation = conformations[np.argmin(energies)]

    # Translate the conformation so the first point is at the origin
    translation = best_conformation[0]
    best_conformation = best_conformation - translation
    
    # Print the lowest conformation achieved (if there is no known lowest or if it was not obtained)
    if E_star is None or E_min > E_star: print(f"Lowest energy obtained: {best_energy}")

    return min(energies), best_conformation, energies_history