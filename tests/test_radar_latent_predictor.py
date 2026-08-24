import pytest

torch = pytest.importorskip("torch")

from models.radar_latent_predictor import LatentSimVPPredictor


def _small_predictor(residual="none"):
    return LatentSimVPPredictor(
        sequence_length=2,
        latent_channels=2,
        latent_height=8,
        latent_width=8,
        hid_s=4,
        hid_t=8,
        n_s=2,
        n_t=2,
        incep_ker=[3],
        groups=2,
        residual=residual,
    )


def test_direct_latent_predictor_shape_and_backward():
    model = _small_predictor(residual="none")
    history = torch.randn(1, 2, 2, 8, 8, requires_grad=True)
    prediction = model(history)
    assert prediction.shape == history.shape
    prediction.square().mean().backward()
    assert history.grad is not None


def test_direct_mode_does_not_add_last_frame():
    direct = _small_predictor(residual="none")
    residual = _small_predictor(residual="last")
    for model in (direct, residual):
        for parameter in model.net.parameters():
            torch.nn.init.zeros_(parameter)
    history = torch.randn(1, 2, 2, 8, 8)
    assert torch.allclose(direct(history), torch.zeros_like(history))
    assert torch.allclose(residual(history), history[:, -1:].expand_as(history))


def test_predictor_rejects_wrong_time_length():
    model = _small_predictor()
    with pytest.raises(ValueError, match="expected T/C"):
        model(torch.randn(1, 3, 2, 8, 8))
