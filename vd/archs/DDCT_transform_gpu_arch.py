import numpy as np
from scipy.fftpack import dct
import torch


def DDCT_transform(im, mode):
    if mode == 0:
        D = DDCTmode0(im)
    elif mode == 1:
        D = DDCTmode0(im.transpose(-1, -2))
    elif mode == 3:
        D = DDCTmode3(im)
    elif mode == 4:
        im = im.permute(1, 2, 0)
        im_T = torch.fliplr(im)
        im_T = im_T.permute(2, 0, 1)
        D = DDCTmode3(im_T)
    elif mode == 5:
        D = DDCTmode5(im)
    elif mode == 6:
        D = DDCTmode5(torch.fliplr(im))
    elif mode == 7:
        D = DDCTmode5(im.transpose(-1, -2))
    elif mode == 8:
        D = DDCTmode5(torch.fliplr(im.transpose(-1, -2)))
    else:
        print('Mode request is not correct, it should be 0 to 8 except 2')
        return None

    return D


def DDCTmode0(im):
    return im


def DDCTmode3(im):
    C, M, N = im.shape
    k = torch.arange(1, 2 * N)

    d = torch.zeros((C, N, len(k)))

    for y in range(len(k)):
        z = 0
        for i in range(N):
            for j in range(N):
                if k[y] == i + j + 1:
                    d[:, z, y] = im[:, i, j]
                    z = z + 1

    return d


def DDCTmode5(im):
    C, M, N = im.shape
    Mh, Nh = M // 2, N // 2
    dd = torch.zeros((C, M, N + 1))
    du = im[:, :Mh, :]
    dl = im[:, Mh:, :]
    dd[:, :Mh, 1:N] = du[:, :, :N - 1]
    dd[:, Mh:, 1:N] = dl[:, :, 1:]
    dd[:, :Mh, 0] = dl[:, :, 0]
    dd[:, :Mh, N] = du[:, :, N - 1]

    return dd
