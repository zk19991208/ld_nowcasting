from .metric import TS, TS_time_, correlation
from .verify import verify_, verify_time_test
import sys
import torch
from PIL import Image
import numpy as np

def Z2R(dBZ, a=200, b=1.6):
    return (10**(dBZ/10.)/a)**(1/b)/10.

def add_score_log_sum(predict_class, predict_class_vmax, seqs_y, decoder_frames, marker="val"):
    metrics_pred = {}
    if 0 in predict_class:
        prec_index = predict_class.index(0)
        seqs_y_uncode_prec = (seqs_y[:, :10, prec_index, :, :] * predict_class_vmax[prec_index]).sum(dim=1)
        decoder_frames_uncode_prec = (decoder_frames[:, :10, prec_index, :, :] * predict_class_vmax[prec_index]).sum(dim=1)
        verify_005mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=0.5, mark_key="1hour_005mm_%s" % marker)
        verify_010mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=1, mark_key="1hour_010mm_%s" % marker)
        verify_050mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=5, mark_key="1hour_050mm_%s" % marker)
        verify_100mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=10, mark_key="1hour_100mm_%s" % marker)

        metrics_pred = {**metrics_pred, **verify_010mm.describe(), **verify_005mm.describe(),
                        **verify_050mm.describe(), **verify_100mm.describe(), }

    return metrics_pred


def add_score_log(predict_class, predict_class_vmax, seqs_y, decoder_frames, marker="val"):
    metrics_pred = {}
    if 0 in predict_class:
        prec_index = predict_class.index(0)
        seqs_y_uncode_prec = seqs_y[:, :, prec_index, :, :] * predict_class_vmax[prec_index]
        decoder_frames_uncode_prec = decoder_frames[:, :, prec_index, :, :] * predict_class_vmax[prec_index]
        radar_index = predict_class.index(1)


        verify_010mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=0.01, mark_key="001mm_%s" % marker)
        verify_050mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=0.5, mark_key="050mm_%s" % marker)
        verify_100mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=1, mark_key="100mm_%s" % marker)
        verify_200mm = verify_(obs=seqs_y_uncode_prec,
                               pred=decoder_frames_uncode_prec,
                               thre_min=2, mark_key="200mm_%s" % marker)

        metrics_pred = {**metrics_pred, **verify_010mm.describe(), **verify_050mm.describe(),
                        **verify_100mm.describe(), **verify_200mm.describe(), }


    if 1 in predict_class:
        radar_index = predict_class.index(1)
        seqs_y_uncode_radar = seqs_y[:, :, radar_index, :, :] * predict_class_vmax[radar_index]
        decoder_frames_uncode_radar = decoder_frames[:, :, radar_index, :, :] * predict_class_vmax[radar_index]
        # print(seqs_y_uncode_radar.shape, decoder_frames_uncode_radar.shape)
        verify_15dBZ = verify_(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=15, mark_key="15dBZ_%s" % marker)
        verify_25dBZ = verify_(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=25, mark_key="25dBZ_%s" % marker)
        verify_35dBZ = verify_(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=35, mark_key="35dBZ_%s" % marker)
        verify_45dBZ = verify_(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=45, mark_key="45dBZ_%s" % marker)


        metrics_pred = {**metrics_pred, **verify_15dBZ.describe(), **verify_25dBZ.describe(),
                        **verify_35dBZ.describe(), **verify_45dBZ.describe()}




    return metrics_pred



def add_time_score_log(predict_class, predict_class_vmax, seqs_y, decoder_frames, marker="test"):
    metrics_pred = {}
    if 0 in predict_class:
        prec_index = predict_class.index(0)
        seqs_y_uncode_prec = seqs_y[:, :, prec_index, :, :] * predict_class_vmax[prec_index]
        decoder_frames_uncode_prec = decoder_frames[:, :, prec_index, :, :] * predict_class_vmax[prec_index]
        verify_010mm_time = verify_time_test(obs=seqs_y_uncode_prec ,
                                             pred=decoder_frames_uncode_prec ,
                                             thre_min=0.01, mark_key="001mm_time_%s"%marker)
        verify_016mm_time = verify_time_test(obs=seqs_y_uncode_prec ,
                                             pred=decoder_frames_uncode_prec ,
                                             thre_min=0.5, mark_key="050mm_time_%s"%marker)

        verify_050mm_time = verify_time_test(obs=seqs_y_uncode_prec,
                                             pred=decoder_frames_uncode_prec,
                                             thre_min=1, mark_key="100mm_time_%s"%marker)

        verify_100mm_time = verify_time_test(obs=seqs_y_uncode_prec,
                                             pred=decoder_frames_uncode_prec,
                                             thre_min=2, mark_key="200mm_time_%s"%marker)
        metrics_pred = {**metrics_pred, **verify_010mm_time.describe(), **verify_016mm_time.describe(),
                        **verify_050mm_time.describe(), **verify_100mm_time.describe(), }
    if 1 in predict_class:
        radar_index = predict_class.index(1)
        seqs_y_uncode_radar = seqs_y[:, :, radar_index, :, :] * predict_class_vmax[radar_index]
        decoder_frames_uncode_radar = decoder_frames[:, :, radar_index, :, :] * predict_class_vmax[radar_index]
        verify_15dB_time = verify_time_test(obs=seqs_y_uncode_radar,
                                            pred=decoder_frames_uncode_radar,
                                            thre_min=15, mark_key="15dBZ_time_%s"%marker)
        verify_25dB_time = verify_time_test(obs=seqs_y_uncode_radar,
                                            pred=decoder_frames_uncode_radar,
                                            thre_min=25, mark_key="25dBZ_time_%s"%marker)
        verify_35dB_time = verify_time_test(obs=seqs_y_uncode_radar ,
                                            pred=decoder_frames_uncode_radar,
                                            thre_min=35, mark_key="35dBZ_time_%s"%marker)
        verify_45dB_time = verify_time_test(obs=seqs_y_uncode_radar,
                                            pred=decoder_frames_uncode_radar,
                                            thre_min=45, mark_key="45dBZ_time_%s"%marker)
        # for i in range(1, 30):
        #     fid = FID()
        #     fid.reset()
        #     fid.process(output=decoder_frames_uncode_radar[:, [0, i], :, :], target=seqs_y_uncode_radar[:, [0, i], :, :])
        #     # # fid.evaluate()
        #     # metrics_pred.update(fid.evaluate())
        #     #
        #     # fid = FID(output=decoder_frames_uncode_radar[:, i, :, :], target=seqs_y_uncode_radar[:, i, :, :])
        #     metrics_pred.update(fid.evaluate())

        metrics_pred = {**metrics_pred, **verify_15dB_time.describe(), **verify_25dB_time.describe(),
                        **verify_35dB_time.describe(), **verify_45dB_time.describe(), }
        # metrics_pred = {**metrics_pred}

    # ZR关系降水评分
    # if 0 in predict_class and 1 in predict_class:
    #     prec_index = predict_class.index(0)
    #     radar_index = predict_class.index(1)
    #     seqs_y_uncode_prec = seqs_y[:, :, prec_index, :, :] * predict_class_vmax[prec_index]
    #     decoder_frames_uncode_radar = decoder_frames[:, :, radar_index, :, :] * predict_class_vmax[radar_index]
    #     ZR_010mm_time = verify_time_test(obs=seqs_y_uncode_prec,
    #                                      pred=Z2R(decoder_frames_uncode_radar),
    #                                      thre_min=0.1, mark_key="010mm_time_zr_%s"%marker)
    #     ZR_016mm_time = verify_time_test(obs=seqs_y_uncode_prec,
    #                                      pred=Z2R(decoder_frames_uncode_radar),
    #                                      thre_min=0.2, mark_key="020mm_time_zr_%s"%marker)
    #
    #     ZR_050mm_time = verify_time_test(obs=seqs_y_uncode_prec,
    #                                      pred=Z2R(decoder_frames_uncode_radar),
    #                                      thre_min=0.5, mark_key="050mm_time_zr_%s"%marker)
    #     ZR_100mm_time = verify_time_test(obs=seqs_y_uncode_prec,
    #                                      pred=Z2R(decoder_frames_uncode_radar),
    #                                      thre_min=1, mark_key="100mm_time_zr_%s"%marker)
    #     metrics_pred = {**metrics_pred, **ZR_010mm_time.describe(), **ZR_016mm_time.describe(),
    #                     **ZR_050mm_time.describe(), **ZR_100mm_time.describe(), }



    if 4 in predict_class:
        wind_index = predict_class.index(4)
        seqs_y_uncode_radar = seqs_y[:, :, wind_index, :, :] * predict_class_vmax[wind_index]
        decoder_frames_uncode_radar = decoder_frames[:, :, wind_index, :, :] * predict_class_vmax[wind_index]
        verify_L5_time = verify_time_test(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=8.0, mark_key="L5_time_%s" % marker)
        verify_L6_time = verify_time_test(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=10.8, mark_key="L6_time_%s" % marker)
        verify_L7_time = verify_time_test(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=13.9, mark_key="L7_time_%s" % marker)
        verify_L8_time = verify_time_test(obs=seqs_y_uncode_radar,
                               pred=decoder_frames_uncode_radar,
                               thre_min=17.2, mark_key="L8_time_%s" % marker)

        metrics_pred = {**metrics_pred, **verify_L5_time.describe(), **verify_L6_time.describe(),
                        **verify_L7_time.describe(), **verify_L8_time.describe(), }
    return metrics_pred

if __name__ =="__main__":
    import sys
    sys.path.append("/home/zhuangxr/work/zengkang/metrics")
    predict_class=[1]
    predict_class_vmax=[70]
    seqs_y = torch.random.rand(1, 3, 64, 64)
    seqs_y = torch.random.rand(1, 3, 64, 64)
    add_score_log(predict_class, predict_class_vmax, seqs_y, decoder_frames, marker="val")