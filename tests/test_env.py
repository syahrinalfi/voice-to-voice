import pytest


@pytest.mark.gpu
def test_cuda_available():
    import torch

    assert torch.cuda.is_available()
