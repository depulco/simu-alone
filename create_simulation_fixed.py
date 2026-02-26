import random
import math
import torch

from simulation import Board, Spike, Food, Boule, Default_spike_pilot, even_spaced_eyes


def _far_enough(x, y, points, min_dist):
    for (px, py) in points:
        if math.hypot(x - px, y - py) < min_dist:
            return False
    return True


def create_sim_test_nn(
    width: int,
    height: int,
    nombre_spikes: int,
    nombre_food: int,
    nombre_boule: int,
    seed: int = None,
    actor_model=None,  # ignoré ici
    min_dist_boule_spike: float = 90.0,   # évite morts instant
    min_dist_food_spike: float = 40.0,    # évite food collée spike
) -> Board:
    """
    Génère une map random.
    - Si seed=None -> seed vraiment aléatoire
    - RNG local (ne touche pas random.seed global)
    """
    if seed is None:
        seed = random.randrange(0, 2**31 - 1)

    rng = random.Random(seed)
    torch.manual_seed(seed)

    board = Board(width, height)

    # ---------------- Boules (d'abord, pour placer les spikes à distance)
    boule_positions = []
    for _ in range(nombre_boule):
        for _try in range(500):
            x = rng.randint(20, width - 20)
            y = rng.randint(20, height - 20)
            if _far_enough(x, y, boule_positions, 30):
                angle = rng.randint(0, 359)
                boule = Boule(x, y, angle, board)
                board.add_boule(boule)
                boule.set_eyes(even_spaced_eyes(8, 0, 140, boule), "food")
                boule.set_eyes(even_spaced_eyes(8, 4, 140, boule), "spike")
                boule.set_pilot(False)  
                boule_positions.append((x, y))
                break

    # ---------------- Spikes random (loin des boules)
    spike_positions = []
    for _ in range(nombre_spikes):
        for _try in range(800):
            x = rng.randint(20, width - 20)
            y = rng.randint(20, height - 20)
            if _far_enough(x, y, spike_positions, 30) and _far_enough(x, y, boule_positions, min_dist_boule_spike):
                pilot = Default_spike_pilot(width, height)
                board.add_spike(Spike(x, y, pilot))
                spike_positions.append((x, y))
                break

    # ---------------- Foods random (pas collés spikes)
    food_positions = []
    for _ in range(nombre_food):
        for _try in range(800):
            x = rng.randint(20, width - 20)
            y = rng.randint(20, height - 20)
            if _far_enough(x, y, food_positions, 20) and _far_enough(x, y, spike_positions, min_dist_food_spike):
                board.add_food(Food(x, y))
                food_positions.append((x, y))
                break

    # petit debug utile si tu veux
    board._seed = seed
    return board