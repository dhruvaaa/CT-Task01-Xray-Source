import numpy as np


N = 100_000                                     
source_position = np.array([600.0, 0.0, 0.0])  
detector_position = np.array([0.0, 0.0, 0.0])   
cone_angle = np.radians(5.0)                   


def sample_photon_energies(energies, spectrum, N):
    """Randomly draw N photon energies according to the spectrum shape."""
    probability = spectrum / spectrum.sum()
    return np.random.choice(energies, size=N, p=probability)


def compute_central_direction(source_position, detector_position):
    """Unit vector pointing from the source toward the detector."""
    direction = detector_position - source_position
    return direction / np.linalg.norm(direction)


def orthonormal_basis(w):
    """Build two unit vectors perpendicular to w (and to each other)."""
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, w)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v


def sample_cone_directions(central_direction, cone_angle, N):
    """Generate N random unit vectors inside a cone around central_direction."""
    cos_theta = np.random.uniform(np.cos(cone_angle), 1.0, N)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = np.random.uniform(0, 2 * np.pi, N)

    u, v = orthonormal_basis(central_direction)

    directions = (
        sin_theta[:, None] * np.cos(phi)[:, None] * u
        + sin_theta[:, None] * np.sin(phi)[:, None] * v
        + cos_theta[:, None] * central_direction
    )
    return directions



photon_energies = sample_photon_energies(energies, spectrum, N)
central_direction = compute_central_direction(source_position, detector_position)
photon_directions = sample_cone_directions(central_direction, cone_angle, N)

