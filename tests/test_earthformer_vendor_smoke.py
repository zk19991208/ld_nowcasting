# 在已安装 torch 的环境下：pytest tests/test_earthformer_vendor_smoke.py -q
# 从 transfer 目录运行；验证 vendored CuboidTransformerModel 可导入且前向形状正确。

import pytest

torch = pytest.importorskip("torch")

from models.earthformer_vendor.earthformer.cuboid_transformer import CuboidTransformerModel


def test_cuboid_transformer_forward_shape():
    m = CuboidTransformerModel(
        input_shape=(10, 64, 64, 1),
        target_shape=(10, 64, 64, 1),
        enc_depth=[4, 4],
        dec_depth=[4, 4],
        enc_attn_patterns=["axial", "axial"],
        dec_self_attn_patterns=["axial", "axial"],
        dec_cross_attn_patterns=["cross_1x1", "cross_1x1"],
        num_global_vectors=0,
        use_dec_self_global=False,
        checkpoint_level=0,
    )
    x = torch.randn(2, 10, 64, 64, 1)
    y = m(x)
    assert y.shape == (2, 10, 64, 64, 1)
