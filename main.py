# %%
import copy

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import integrate

# Start and end of the period
x0: float = 0
xe: float = 10

# Number of terms in the fourier series
N: int = 20

x: npt.NDArray[np.float64] = np.arange(x0, xe, 0.001, dtype=np.float64)
y: npt.NDArray[np.float64] = np.exp(-0.2 * x) * np.cos(3 * x)

P: float = xe - x0
omega: float = 2 * np.pi / P


def calculate_score(y: npt.NDArray[np.float64], fit: npt.NDArray[np.float64], P: float) -> float:
    """Calculate the score, which is the Root Mean Square Error"""
    return np.sqrt(np.sum((y - fit)**2) / P)


def generate_genes(number_of_parents: int, N: int) -> npt.NDArray[np.float64]:
    bounds: tuple[float, float] = (-0.3, 0.3)
    return np.random.uniform(low=bounds[0], high=bounds[1], size=(number_of_parents, 2 * N))


def crossover(parent_1: npt.NDArray[np.float64], parent_2: npt.NDArray[np.float64]):
    child_1: npt.NDArray[np.float64] = copy.deepcopy(parent_1)
    child_2: npt.NDArray[np.float64] = copy.deepcopy(parent_2)

    crossover_idx: int = np.random.randint(len(parent_1) + 1)

    child_1[crossover_idx] = parent_2[crossover_idx]
    child_2[crossover_idx] = parent_1[crossover_idx]

    return child_1, child_2


def selection(generation: npt.NDArray[np.float64], scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pick two parents based on the K-way tournament selection. """
    participant_idxs: npt.NDArray[np.int64] = np.random.randint(0, generation.shape[0] + 1, size=(3,))

    while not np.unique(participant_idxs).shape[0] == participant_idxs.shape[0]:
        participant_idxs: npt.NDArray[np.int64] = np.random.randint(0, generation.shape[0] + 1, size=(3,))

    participant_scores = scores[participant_idxs]
    best_score = np.min(participant_scores)
    idx = np.where(participant_scores == best_score)

    parent = generation[idx]
    return parent


def generate_fourier_series(
    x: npt.NDArray[np.float64],
    genes: npt.NDArray[np.float64],
    omega: float,
    N: int
) -> npt.NDArray[np.float64]:

    a_n = genes[:N]
    b_n = genes[N:]

    z: tuple[float] = integrate.quad(lambda x: 1 + np.exp(-0.2 * x) * np.cos(3 * x), 0, P)
    a_0 = 2 / P * z[0]

    fit: npt.NDArray[np.float64] = a_0 / 2 + np.zeros(len(x))

    for n in range(len(a_n)):
        fit += a_n[n] * np.cos(omega * n * x) + b_n[n] * np.sin(omega * n * x)

    return fit


number_of_parents: int = 100

initial_generation = generate_genes(number_of_parents, N)
scores: npt.NDArray[np.float64] = np.zeros(number_of_parents)
for i in range(100):
    fit = generate_fourier_series(x, initial_generation[i], omega, N)
    scores[i] = calculate_score(y, fit, P)

new_generation = np.zeros()

parent_1 = selection(initial_generation, scores)
parent_2 = selection(initial_generation, scores)
child_1, child_2 = crossover(parent_1, parent_2)

print('mean', np.mean(scores))

# plt.figure()
# plt.plot(x, fit)
# plt.plot(x, y)
# plt.show()

# %%
