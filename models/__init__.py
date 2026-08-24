# 只导出当前训练入口实际使用且兼容 Lightning 2.x 的模型。
# 旧版 PhyDNet/PredRNN 仍应从各自子模块显式导入。

from .simvp import SimVP_Lit
from .radar_autoencoder import RadarAutoEncoder, RadarAutoEncoderLit
from .radar_latent_predictor import LatentSimVPPredictor, RadarLatentPredictorLit

__all__ = [
    "SimVP_Lit",
    "RadarAutoEncoder",
    "RadarAutoEncoderLit",
    "LatentSimVPPredictor",
    "RadarLatentPredictorLit",
]
