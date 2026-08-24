import pytest

torch = pytest.importorskip("torch")

from loss.radar_reconstruction import RadarReconstructionLoss
from models.radar_autoencoder import RadarAutoEncoder


@pytest.mark.parametrize(
    ("channels", "factor", "latent_hw"),
    [((16, 32, 64), 4, 16), ((16, 32, 64, 64), 8, 8)],
)
def test_radar_autoencoder_shape_and_backward(channels, factor, latent_hw):
    model = RadarAutoEncoder(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        block_out_channels=channels,
        layers_per_block=1,
        norm_groups=8,
    )
    x = torch.rand(2, 1, 64, 64, requires_grad=True)
    z = model.encode(x)
    y = model.decode(z)
    assert model.downsample_factor == factor
    assert z.shape == (2, 4, latent_hw, latent_hw)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None


def test_radar_autoencoder_rejects_non_divisible_shape():
    model = RadarAutoEncoder(block_out_channels=(16, 32, 64, 64), layers_per_block=1)
    with pytest.raises(ValueError, match="divisible"):
        model.encode(torch.rand(1, 1, 62, 64))


def test_radar_reconstruction_loss_prefers_exact_reconstruction():
    loss_fn = RadarReconstructionLoss()
    target = torch.rand(2, 1, 32, 32)
    exact, _ = loss_fn(target, target)
    wrong, parts = loss_fn(torch.zeros_like(target), target)
    assert exact.item() == pytest.approx(0.0)
    assert wrong > exact
    assert set(parts) == {"weighted_l1", "mse", "gradient", "total"}
