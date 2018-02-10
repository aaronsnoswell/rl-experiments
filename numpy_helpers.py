
import numpy as np
from contextlib import contextmanager

@contextmanager
def show_complete_array():
    """
    Shows a complete numpy array without truncation
    From https://stackoverflow.com/a/45831462/885287
    """
    oldoptions = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    yield
    np.set_printoptions(**oldoptions)

