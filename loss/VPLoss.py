import torch
from torch import nn

e = 2.718281828459045
pi = 3.141592653589793
theta2_prec = 1.134620682038696
theta2_radar = 310.28654450478393

def half_normal_weight_func(x, theta2):
    weight = (pi * theta2 / 2.) ** 0.5 * e ** (x ** 2 / (2 * theta2))
    return torch.where(weight > 30, torch.full_like(weight, 30.), weight)

class weighted_mae_windows(nn.Module):
    def __init__(self, weights=(1, 1, 2.5, 5, 10, 20),
                 thresholds=(0.0125, 0.025, 0.0625, 0.125, 0.25)):
        super(weighted_mae_windows, self).__init__()
        assert len(thresholds) + 1 == len(weights)
        self.weights = weights
        self.threholds = thresholds

    def forward(self, predict, target):
        """
        :param input: nbatchs * nlengths * nheigths * nwidths
        :param target: nbatchs * nlengths * nheigths * nwidths
        :return:
        """
        balance_weights = torch.zeros_like(target)
        balance_weights[target < self.threholds[0]] = self.weights[0]
        for i, _ in enumerate(self.threholds[:-1]):
            balance_weights[(target >= self.threholds[i]) & (target < self.threholds[i + 1])] = self.weights[i + 1]
        balance_weights[target >= self.threholds[-1]] = self.weights[-1]
        mae = torch.mean(balance_weights * (torch.abs(predict - target)))
        return mae

class weighted_mae_windows_wind(nn.Module):
    def __init__(self, speed_index, weights=(1, 1, 2.5, 5, 10, 20),
                 thresholds=(0.0125, 0.025, 0.0625, 0.125, 0.25),):
        super(weighted_mae_windows_wind, self).__init__()
        assert len(thresholds) + 1 == len(weights)
        self.weights = weights
        self.threholds = thresholds
        self.speed_index = speed_index

    def forward(self, predict, target):
        """
        :param input: nbatchs * nlengths * 3 * nheigths * nwidths
        :param target: nbatchs * nlengths * 3 * nheigths * nwidths
        :return:
        """
        target_speed = target[:, :, self.speed_index:self.speed_index + 1, ...]
        balance_weights = torch.zeros_like(target_speed)
        balance_weights[target_speed < self.threholds[0]] = self.weights[0]
        for i, _ in enumerate(self.threholds[:-1]):
            balance_weights[(target_speed >= self.threholds[i]) & (target_speed < self.threholds[i + 1])] = self.weights[i + 1]
        balance_weights[target_speed >= self.threholds[-1]] = self.weights[-1]
        mae = torch.mean(balance_weights * (torch.abs(predict - target)))
        return mae


class weighted_mse_windows(nn.Module):
    def __init__(self, weights=(0.5, 1, 2.5, 5, 10, 15),
                 thresholds=(0.21, 0.36, 0.5, 0.57, 0.64), ):
        super(weighted_mse_windows, self).__init__()
        assert len(thresholds) + 1 == len(weights)
        self.weights = weights
        self.threholds = thresholds

    def forward(self, predict, target):
        """
        :param input: nbatchs * nlengths * nheigths * nwidths
        :param target: nbatchs * nlengths * nheigths * nwidths
        :return:
        """
        balance_weights = torch.zeros_like(target)
        balance_weights[target < self.threholds[0]] = self.weights[0]
        for i, _ in enumerate(self.threholds[:-1]):
            balance_weights[(target >= self.threholds[i]) & (target < self.threholds[i + 1])] = self.weights[i + 1]
        balance_weights[target >= self.threholds[-1]] = self.weights[-1]
        mse = torch.mean(balance_weights * (torch.square(predict - target)))
        return mse


class weighted_l1_radar(nn.Module):
    def __init__(self, weights_radar, thresholds_radar, unnorm_radar):
        super(weighted_l1_radar, self).__init__()
        thresholds_radar_code = [iradar / unnorm_radar for iradar in thresholds_radar]
        self.weighted_radar = weighted_mae_windows(weights=weights_radar,
                                                   thresholds=thresholds_radar_code)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 1 * nheigths * nwidths
        :param target: nbatchs * nlengths * 1 * nheigths * nwidths
        :return:
        """
        mae = self.weighted_radar(output, target)
        return mae


class weighted_l2_radar(nn.Module):
    def __init__(self, weights_radar, thresholds_radar, unnorm_radar):
        super(weighted_l2_radar, self).__init__()
        thresholds_radar_code = [iradar / unnorm_radar for iradar in thresholds_radar]
        self.weighted_radar = weighted_mse_windows(weights=weights_radar,
                                                   thresholds=thresholds_radar_code)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 1 * nheigths * nwidths
        :param target: nbatchs * nlengths * 1 * nheigths * nwidths
        :return:
        """
        mse = self.weighted_radar(output, target)
        return mse



class weighted_l1_prec(nn.Module):
    def __init__(self, weights_prec, thresholds_prec, unnorm_prec):
        super(weighted_l1_prec, self).__init__()
        thresholds_prec_code = [iprec / unnorm_prec for iprec in thresholds_prec]
        self.weighted_prec = weighted_mae_windows(weights=weights_prec,
                                                  thresholds=thresholds_prec_code)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 1 * nheigths * nwidths
        :param target: nbatchs * nlengths * 1 * nheigths * nwidths
        :return:
        """
        mae = self.weighted_prec(output, target)
        return mae


class weighted_l2_prec(nn.Module):
    def __init__(self, weights_prec, thresholds_prec, unnorm_prec):
        super(weighted_l2_prec, self).__init__()
        thresholds_prec_code = [iprec / unnorm_prec for iprec in thresholds_prec]
        self.weighted_prec = weighted_mse_windows(weights=weights_prec,
                                                  thresholds=thresholds_prec_code)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 1 * nheigths * nwidths
        :param target: nbatchs * nlengths * 1 * nheigths * nwidths
        :return:
        """
        mse = self.weighted_prec(output, target)
        return mse

    

class weighted_l1_c2(nn.Module):
    def __init__(self, weights_prec, thresholds_prec, weights_radar, thresholds_radar,
                 unnorm_prec, unnorm_radar, predict_class=[0, 1]):
        super(weighted_l1_c2, self).__init__()
        thresholds_prec_code = [iprec / unnorm_prec for iprec in thresholds_prec]
        thresholds_radar_code = [iradar / unnorm_radar for iradar in thresholds_radar]
        self.predict_class = predict_class
        self.index_prec = predict_class.index(0)
        self.index_radar = predict_class.index(1)

        self.weighted_prec = weighted_mae_windows(weights=weights_prec,
                                                  thresholds=thresholds_prec_code)
        self.weighted_radar = weighted_mae_windows(weights=weights_radar,
                                                   thresholds=thresholds_radar_code)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """
        assert output.shape[2] == 2, "the channel must be 2!"
        mae = self.weighted_prec(output[:, :, self.index_prec, ...], target[:, :, self.index_prec, ...]) + \
              self.weighted_radar(output[:, :, self.index_radar, ...], target[:, :, self.index_radar, ...])
        return mae

class weighted_l2_c2(nn.Module):
    def __init__(self, weights_prec, thresholds_prec, weights_radar, thresholds_radar,
                 unnorm_prec, unnorm_radar, predict_class=[0, 1]):
        super(weighted_l2_c2, self).__init__()
        thresholds_prec_code = [iprec / unnorm_prec for iprec in thresholds_prec]
        thresholds_radar_code = [iradar / unnorm_radar for iradar in thresholds_radar]
        self.predict_class = predict_class
        self.index_prec = predict_class.index(0)
        self.index_radar = predict_class.index(1)

        self.weighted_prec = weighted_mse_windows(weights=weights_prec,
                                                  thresholds=thresholds_prec_code)
        self.weighted_radar = weighted_mse_windows(weights=weights_radar,
                                                   thresholds=thresholds_radar_code)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """
        assert output.shape[2] == 2, "the channel must be 2!"
        mse = self.weighted_prec(output[:, :, self.index_prec, ...], target[:, :, self.index_prec, ...]) + \
              self.weighted_radar(output[:, :, self.index_radar, ...], target[:, :, self.index_radar, ...])
        return mse


class NormalDisWeight_l1_c2(nn.Module):

    def __init__(self, unnorm_prec, unnorm_radar, predict_class=[0, 1]):
        super(NormalDisWeight_l1_c2, self).__init__()
        self.theta2_prec = theta2_prec / (unnorm_prec ** 2)
        self.theta2_radar = theta2_radar / (unnorm_radar ** 2)
        self.predict_class = predict_class
        self.index_prec = predict_class.index(0)
        self.index_radar = predict_class.index(1)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """

        prec_weight = half_normal_weight_func(target[:, :, self.index_prec, :, :], self.theta2_prec)
        radar_weight = half_normal_weight_func(target[:, :, self.index_radar, :, :], self.theta2_radar)
        mae = torch.mean(
            prec_weight * (torch.abs(output[:, :, self.index_prec, :, :] - target[:, :, self.index_prec, :, :]))) + \
              torch.mean(radar_weight * (
                  torch.abs(output[:, :, self.index_radar, :, :] - target[:, :, self.index_radar, :, :])))
        return mae


class NormalDisWeight_l2_c2(nn.Module):

    def __init__(self, unnorm_prec, unnorm_radar, predict_class=[0, 1]):
        super(NormalDisWeight_l2_c2, self).__init__()
        self.theta2_prec = theta2_prec / (unnorm_prec ** 2)
        self.theta2_radar = theta2_radar / (unnorm_radar ** 2)
        self.predict_class = predict_class
        self.index_prec = predict_class.index(0)
        self.index_radar = predict_class.index(1)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """
        prec_weight = half_normal_weight_func(target[:, :, self.index_prec, :, :], self.theta2_prec)
        radar_weight = half_normal_weight_func(target[:, :, self.index_radar, :, :], self.theta2_radar)
        mse = torch.mean(
            prec_weight * (torch.square(output[:, :, self.index_prec, :, :] - target[:, :, self.index_prec, :, :]))) + \
              torch.mean(radar_weight * (
                  torch.square(output[:, :, self.index_radar, :, :] - target[:, :, self.index_radar, :, :])))
        return mse


class NormalDisWeight_l1_radar(nn.Module):

    def __init__(self, unnorm_radar):
        super(NormalDisWeight_l1_radar, self).__init__()
        self.theta2_radar = theta2_radar / (unnorm_radar ** 2)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """

        radar_weight = half_normal_weight_func(target, self.theta2_radar)
        mae = torch.mean(radar_weight * (torch.abs(output - target)))
        return mae


class NormalDisWeight_l2_radar(nn.Module):

    def __init__(self, unnorm_radar):
        super(NormalDisWeight_l2_radar, self).__init__()
        self.theta2_radar = theta2_radar / (unnorm_radar ** 2)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """

        radar_weight = half_normal_weight_func(target, self.theta2_radar)
        mse = torch.mean(radar_weight * (torch.square(output - target)))
        return mse


class NormalDisWeight_l1_prec(nn.Module):

    def __init__(self, unnorm_prec):
        super(NormalDisWeight_l1_prec, self).__init__()
        self.theta2_prec = theta2_prec / (unnorm_prec ** 2)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """

        prec_weight = half_normal_weight_func(target, self.theta2_prec)
        mae = torch.mean(prec_weight * (torch.abs(output - target)))
        return mae


class NormalDisWeight_l2_prec(nn.Module):

    def __init__(self, unnorm_prec):
        super(NormalDisWeight_l2_prec, self).__init__()
        self.theta2_prec = theta2_prec / (unnorm_prec ** 2)

    def forward(self, output, target):
        """
        :param input: nbatchs * nlengths * 2 * nheigths * nwidths
        :param target: nbatchs * nlengths * 2 * nheigths * nwidths
        :return:
        """
        prec_weight = half_normal_weight_func(target, self.theta2_prec)
        mse = torch.mean(prec_weight * (torch.square(output - target)))
        return mse


        return mae