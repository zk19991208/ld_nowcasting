from math import factorial

import torch
import torch.nn as nn
from torch.linalg import inv


def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats) + 1
    k = x.dim() - 1
    for i in range(k):
        x = torch.tensordot(mats[k - i - 1], x, dims=[[1], [k]])
    x = torch.movedim(x, k, 0)
    return x


def _apply_axis_right_dot(x, mats):
    assert x.dim() == len(mats) + 1
    k = x.dim() - 1
    x = torch.movedim(x, 0, k)
    for i in range(k):
        x = torch.tensordot(x, mats[i], dims=[[0], [0]])
    return x


class K2M(nn.Module):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float32)
        m = k2m(k)
    """

    def __init__(self, shape, device=None):
        super(K2M, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        self.device = device

        for index, l in enumerate(shape):
            M = torch.zeros((l, l), dtype=torch.float32, device=self.device)
            for i in range(l):
                M[i, :] = ((torch.arange(l, device=self.device) - (l - 1) // 2) ** i) / factorial(i)
            self.register_buffer('_M' + str(index), M)

    @property
    def M(self):
        return list(self._buffers['_M' + str(j)] for j in range(self.dim()))

    def size(self):
        return self._size

    def dim(self):
        return self._dim

    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        k = _apply_axis_left_dot(k, self.M)
        return k


class M2K(nn.Module):
    """
    convert moment matrix to convolution kernel
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        m2k = M2K([5,5])
        m = torch.randn(5,5,dtype=torch.float32)
        k = m2k(m)
    """

    def __init__(self, shape, device=None):
        super(M2K, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        self.device = device

        for index, l in enumerate(shape):
            M = torch.zeros((l, l), dtype=torch.float32, device=self.device)
            for i in range(l):
                M[i, :] = ((torch.arange(l, device=self.device) - (l - 1) // 2) ** i) / factorial(i)
            self.register_buffer('_invM' + str(index), inv(M))

    @property
    def invM(self):
        return list(self._buffers['_invM' + str(j)] for j in range(self.dim()))

    def size(self):
        return self._size

    def dim(self):
        return self._dim

    def forward(self, m):
        """
        m (Tensor): torch.size=[...,*self.shape]
        """
        m = _apply_axis_left_dot(m, self.invM)
        return m
