import numpy as np

GRID_POINTS_PER_SIDE = 29
GRID_SPACING = 0.1
SCALE_FACTOR = 1e6


def compute_detector_matrix(detector_depth: float = 1.0, aperture: float = 0.02) -> np.ndarray:

    coords = GRID_SPACING * np.arange(GRID_POINTS_PER_SIDE)
    xs, ys = np.meshgrid(coords, coords, indexing="ij")
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    dx = xs - xs.T
    dy = ys - ys.T
    dz = detector_depth
    rnorm = np.sqrt(dx**2 + dy**2 + dz**2)
    kernel = np.exp(-((dz / rnorm - 1) ** 2) / (2 * aperture**2)) / (
        np.sqrt(2 * np.pi) * aperture
    )/3

    return kernel * (GRID_SPACING**2)


def generate_detector_measurement(
    ideal_measurement: np.ndarray, noise_level: float, detector_matrix: np.ndarray
) -> np.ndarray:
    ideal0 = ideal_measurement[0]
    ideal = ideal_measurement[1:] * SCALE_FACTOR
    n = GRID_POINTS_PER_SIDE
    flat = n * n
    detector = ideal.reshape(-1, flat) @ detector_matrix
    detector = detector.reshape(-1, n, n, n, n)
    white_noise = np.random.normal(size=ideal.shape)
    noise_scale = np.sqrt(np.clip(ideal0, 0, None)).reshape(1,n,n,n,n)
    detector = detector + noise_level * noise_scale * white_noise
    return np.clip(detector, 0, np.inf)
