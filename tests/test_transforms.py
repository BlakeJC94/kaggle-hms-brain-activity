import torch
import pytest

from hms_brain_activity import transforms as t
from hms_brain_activity.globals import CHANNEL_NAMES


class MockMontage(t._BaseMontage):
    montage = [("Fp1", "F3"), ("P3", "C3"), ("F7", "")]  # 0, 1  # 3, 2  # 4


@pytest.mark.parametrize(
    "shape",
    [
        (len(CHANNEL_NAMES), 30),  # Sample
        (4, len(CHANNEL_NAMES), 30),  # Batch
    ],
)
def test_base_montage(shape):
    """Test that the _BaseMontage.forward method works as expected."""
    ## Assemble
    transform = MockMontage()
    data = torch.rand(len(CHANNEL_NAMES), 30)  # channels, timesteps
    metadata = dict(foo="bar")

    ## Act
    output, md = transform(data, metadata)

    ## Assert
    assert md == metadata
    assert (output[0, :] == data[0, :] - data[1, :]).all()
    assert (output[1, :] == data[3, :] - data[2, :]).all()
    assert (output[2, :] == data[4, :]).all()


@pytest.mark.parametrize(
    "shape",
    [
        (len(CHANNEL_NAMES), 30),  # Sample
        (4, len(CHANNEL_NAMES), 30),  # Batch
    ],
)
def test_random_scale(shape):
    """Test that the RandomScale.forward method works as expected."""
    ## Assemble
    transform = t.RandomScale()
    data = torch.rand(shape)
    metadata = dict(foo="bar")

    ## Act
    output, md = transform(data, metadata)

    ## Assert
    assert md == metadata
    assert output.ndim == data.ndim
    assert (output < transform.max_scale * data).all()
    assert (output > transform.min_scale * data).all()
