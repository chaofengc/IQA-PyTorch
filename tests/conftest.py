import pytest
import torch



@pytest.fixture(scope='module')
def device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
