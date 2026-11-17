
import random
import numpy as np
import torch


def set_seed(seed, deterministic=True):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed Python and NumPy inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    """Create a seeded PyTorch DataLoader generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
