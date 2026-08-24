import torch


def rain_rmse(output, target):
    rmse = torch.sqrt(torch.mean(torch.square(output - target)))
    return rmse


def rain_mae(output, target):
    mae = torch.mean(torch.abs(output - target))
    return mae


def rain_rmse_time_(output, target):
    rmse = torch.sqrt(torch.mean(torch.square(output - target), axis=(0, 2, 3, 4)))
    return rmse


def rain_mae_time_(output, target):
    mae = torch.mean(torch.abs(output - target), axis=(0, 2, 3, 4))
    return mae


def HSS(output, target, thres):
    a, c, b, d = get_hit_miss_counts(output, target, thres)
    n = a + b + c + d  ##总样本数
    return ((a + d) / n - ((a + b) * (a + c) + (b + d) * (c + d)) / n ** 2) / \
           (1 - ((a + b) * (a + c) + (b + d) * (c + d)) / n ** 2)


def TS(output, target, thres):
    a, c, b, d = get_hit_miss_counts(output, target, thres)
    n = a + b + c  ##总样本数
    return a / (n + 1e-8)


def get_hit_miss_counts(output, target, thresholds=0.2):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:
    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1
    Parameters
    ----------
    prediction : torch.ndarray
        Shape: (batch_size, seq_len,  1, height, width)
    truth : torch.ndarray
        Shape: (batch_size, seq_len,  1, height, width)
    mask : torch.ndarray or None
        Shape: (batch_size, seq_len,  1, height, width)
        0 --> not use
        1 --> use
    thresholds : scalar
    Returns
    -------
    hits : scalar
        TP
    misses : scalar
        FN
    false_alarms : scalar
        FP
    correct_negatives : scalar
        TN
    """
    assert output.shape == target.shape

    bpred = (output >= thresholds)
    btruth = (target >= thresholds)
    bpred_n = torch.logical_not(bpred)
    btruth_n = torch.logical_not(btruth)

    hits = torch.logical_and(bpred, btruth).sum()
    misses = torch.logical_and(bpred_n, btruth).sum()
    false_alarms = torch.logical_and(bpred, btruth_n).sum()
    correct_negatives = torch.logical_and(bpred_n, btruth_n).sum()
    return hits, misses, false_alarms, correct_negatives


def HSS_time_(output, target, thres):
    a, c, b, d = get_hit_miss_counts_time(output, target, thres)
    n = a + b + c + d  ##总样本数
    return ((a + d) / n - ((a + b) * (a + c) + (b + d) * (c + d)) / n ** 2) / \
           (1 - ((a + b) * (a + c) + (b + d) * (c + d)) / n ** 2)


def TS_time_(output, target, thres):
    a, c, b, d = get_hit_miss_counts_time(output, target, thres)
    n = a + b + c  ##总样本数
    return a / (n + 1e-8)


def get_hit_miss_counts_time(output, target, thresholds=0.2):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:
    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1
    Parameters
    ----------
    prediction : torch.ndarray
        Shape: (batch_size, seq_len,  1, height, width)
    truth : torch.ndarray
        Shape: (batch_size, seq_len,  1, height, width)
    thresholds : scalar
    Returns
    -------
    hits : scalar
        TP
    misses : scalar
        FN
    false_alarms : scalar
        FP
    correct_negatives : scalar
        TN
    """
    # assert 5 == output.ndim
    # assert 5 == target.ndim
    # assert output.shape == target.shape
    # assert output.shape[2] == 1

    bpred = (output >= thresholds)
    btruth = (target >= thresholds)
    bpred_n = torch.logical_not(bpred)
    btruth_n = torch.logical_not(btruth)

    hits = torch.logical_and(bpred, btruth).sum(axis=(0, 2, 3, 4))
    misses = torch.logical_and(bpred_n, btruth).sum(axis=(0, 2, 3, 4))
    false_alarms = torch.logical_and(bpred, btruth_n).sum(axis=(0, 2, 3, 4))
    correct_negatives = torch.logical_and(bpred_n, btruth_n).sum(axis=(0, 2, 3, 4))
    return hits, misses, false_alarms, correct_negatives


def correlation(output, target):
    """
    Parameters
    ----------
    prediction : torch.ndarray
    truth : torch.ndarray
    Returns
    -------
    """
    assert output.shape == target.shape
    # assert 5 == output.ndim
    # assert output.shape[2] == 1
    eps = 1E-12
    ret = (output * target).sum(axis=(-2, -1)) / (
            torch.sqrt(torch.square(output).sum(axis=(-2, -1))) * torch.sqrt(torch.square(target).sum(axis=(-2, -1))) + eps)
    ret = ret.mean()
    return ret


def correlation_time_(output, target):
    """
    Parameters
    ----------
    prediction : torch.ndarray
    truth : torch.ndarray
    Returns
    -------
    """
    assert output.shape == target.shape
    assert 5 == output.ndim
    assert output.shape[2] == 1
    eps = 1E-12
    ret = (output * target).sum(axis=(3, 4)) / (
            torch.sqrt(torch.square(output).sum(axis=(3, 4))) * torch.sqrt(torch.square(target).sum(axis=(3, 4))) + eps)
    ret = ret.mean(axis=(0, 2))
    return ret
