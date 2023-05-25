from typing import List
import pytest
import torch
import os
import pandas as pd
import numpy as np

from pyiqa.utils import imread2tensor


@pytest.fixture(scope='module')
def device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

