from torch import nn

from .VPLoss import weighted_l1_c2, weighted_l2_c2, weighted_l1_prec, weighted_l2_prec, weighted_l1_radar, \
    weighted_l2_radar, NormalDisWeight_l1_radar, NormalDisWeight_l1_prec, NormalDisWeight_l1_c2, NormalDisWeight_l2_c2, \
    NormalDisWeight_l2_prec, NormalDisWeight_l2_radar

def get_loss_fx(loss_fx_str, predict_class, predict_class_vmax, weights_prec, thresholds_prec, weights_radar,
                thresholds_radar):
    if loss_fx_str == "weighted_l1":
        if len(predict_class) == 2:
            if (0 in predict_class) and (1 in predict_class):
                prec_index = predict_class.index(0)
                radar_index = predict_class.index(1)
                loss_fx = weighted_l1_c2(weights_prec, thresholds_prec, weights_radar, thresholds_radar,
                                         predict_class_vmax[prec_index], predict_class_vmax[radar_index], predict_class)
            else:
                raise ("NotImplemented!")
        elif len(predict_class) == 1:
            if (0 in predict_class) or (10 in predict_class):
                loss_fx = weighted_l1_prec(weights_prec, thresholds_prec, predict_class_vmax[0])
            elif 1 in predict_class:
                loss_fx = weighted_l1_radar(weights_radar, thresholds_radar, predict_class_vmax[0])

            else:
                raise ("the predict class occur error!")
        else:
            raise ("NotImplemented!")
    elif loss_fx_str == "weight_l2":
        if len(predict_class) == 2:
            if (0 in predict_class) and (1 in predict_class):
                prec_index = predict_class.index(0)
                radar_index = predict_class.index(1)
                loss_fx = weighted_l2_c2(weights_prec, thresholds_prec, weights_radar, thresholds_radar,
                                         predict_class_vmax[prec_index], predict_class_vmax[radar_index], predict_class)
            else:
                raise ("NotImplemented!")
        elif len(predict_class) == 1:
            if 0 in predict_class:
                loss_fx = weighted_l2_prec(weights_prec, thresholds_prec, predict_class_vmax[0])
            elif 1 in predict_class:
                loss_fx = weighted_l2_radar(weights_radar, thresholds_radar, predict_class_vmax[0])
            else:
                raise ("the predict class occur error!")
        else:
            raise ("NotImplemented!")

    elif loss_fx_str == "normal_l1":
        if len(predict_class) == 2:
            if (0 in predict_class) and (1 in predict_class):
                prec_index = predict_class.index(0)
                radar_index = predict_class.index(1)
                loss_fx = NormalDisWeight_l1_c2(predict_class_vmax[prec_index], predict_class_vmax[radar_index],
                                                predict_class)
            else:
                raise ("NotImplemented!")
        elif len(predict_class) == 1:
            if 0 in predict_class:
                loss_fx = NormalDisWeight_l1_prec(predict_class_vmax[0])
            elif 1 in predict_class:
                loss_fx = NormalDisWeight_l1_radar(predict_class_vmax[0])
            else:
                raise ("the predict class occur error!")
        else:
            raise ("NotImplemented!")
    elif loss_fx_str == "normal_l2":
        if len(predict_class) == 2:
            if (0 in predict_class) and (1 in predict_class):
                prec_index = predict_class.index(0)
                radar_index = predict_class.index(1)
                loss_fx = NormalDisWeight_l2_c2(predict_class_vmax[prec_index], predict_class_vmax[radar_index],
                                                predict_class)
            else:
                raise ("NotImplemented!")
        elif len(predict_class) == 1:
            if 0 in predict_class:
                loss_fx = NormalDisWeight_l2_prec(predict_class_vmax[0])
            elif 1 in predict_class:
                loss_fx = NormalDisWeight_l2_radar(predict_class_vmax[0])
            else:
                raise ("the predict class occur error!")
        else:
            raise ("NotImplemented!")
    elif loss_fx_str == "l1":
        loss_fx = nn.L1Loss()
    elif loss_fx_str == "l2":
        loss_fx = nn.MSELoss()
    else:
        raise ("the loss not in 'l1 l2 weighted_l1 weighted l2 normal_l1, normal_l2'!")
    return loss_fx
