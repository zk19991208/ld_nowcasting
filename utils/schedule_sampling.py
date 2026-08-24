import math
import random

import torch


def reserve_schedule_sampling_exp(itr, r_sampling_step_1, r_sampling_step_2, r_exp_alpha, batch_size,
                                  input_length, target_length, device):
    if itr < r_sampling_step_1:
        r_eta = 0.5
    elif itr < r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - r_sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < r_sampling_step_1:
        eta = 0.5
    elif itr < r_sampling_step_2:
        eta = 0.5 - (0.5 / (r_sampling_step_2 - r_sampling_step_1)) * (itr - r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = torch.rand(
        (batch_size, input_length - 1), device=device)
    r_true_token = (r_random_flip < r_eta)

    random_flip = torch.rand(
        (batch_size, target_length - 1), device=device)
    true_token = (random_flip < eta)

    r_true_token_input = r_true_token[..., None, None, None]
    true_token_target = true_token[..., None, None, None]

    return r_true_token_input.float(), true_token_target.float(), r_eta, eta


def schedule_sampling(itr, sampling_stop_iter, sampling_changing_rate,
                      batch_size, input_length, target_length, device):
    if itr < sampling_stop_iter:
        eta = 1.0 - itr * sampling_changing_rate
    else:
        eta = 0.0
    random_flip = torch.rand(
        (batch_size, target_length - 1), device=device)
    true_token = (random_flip < eta)

    return torch.ones(1, input_length - 1, 1, 1, 1, device=device), true_token[..., None, None, None].float(), 1, eta


def teacher_forcing(epoch, sampling_changing_rate_epoch=0.003):
    teacher_forcing_rate = max(0, 1 - epoch * sampling_changing_rate_epoch)
    return True if random.random() < teacher_forcing_rate else False, teacher_forcing_rate
