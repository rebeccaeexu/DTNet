import numpy as np
# from scipy.fftpack import idct
import torch


def IDDCT(IM, mode):
    # compute different inverse DDCT modes
    # VPS Naidu, vpsnaidu@gmail.com

    if mode == 0:
        d = IDDCTmode0(IM)
    elif mode == 1:
        d = IDDCTmode0(IM.transpose(-1, -2))
    elif mode == 3:
        d = IDDCTmode3(IM)
    elif mode == 4:
        dd = IDDCTmode3(IM)
        d = torch.fliplr(dd)
    elif mode == 5:
        d = IDDCTmode5(IM)
    elif mode == 6:
        d = IDDCTmode5(IM)
        d = torch.fliplr(d)
    elif mode == 7:
        d = IDDCTmode5(IM)
        d = d.transpose(-1, -2)
    elif mode == 8:
        d = IDDCTmode5(IM)
        d = torch.fliplr(d)
        d = d.transpose(-1, -2)
    else:
        print('Mode request is not correct, it should be 0 to 8 except 2')
        return None

    return d


def IDDCTmode0(im):
    C, M, N = im.shape
    DD = torch.zeros((C, M, N))
    D = torch.zeros((C, M, N))

    for i in range(N):
        DD[:, :, i] = torch.fft.ifft(im[:, :, i], norm='ortho', dim=-1).real

    for j in range(M):
        D[:, j, :] = torch.fft.ifft(DD[:, j, :], norm='ortho', dim=-1).real
    return D


def IDDCTmode3(D):
    C, M, N = D.shape

    # create Nk tensor with concatenation of two range tensors
    Nk1 = torch.arange(1, M + 1, dtype=torch.float32)
    Nk2 = torch.arange(M - 1, 0, -1, dtype=torch.float32)
    Nk = torch.cat((Nk1, Nk2))

    # create zero tensor dd with size (M, len(Nk))
    dd = torch.zeros((C, M, len(Nk)))

    for i in range(M):
        xx = D[:, i, 0:N - 2 * i]
        dd[:, i, i:N - i] = torch.fft.ifft(xx, norm='ortho', dim=-1).real

    d = torch.zeros((C, M, len(Nk)))

    for i in range(N):
        x = dd[:, 0:int(Nk[i]), i]
        d[:, 0:int(Nk[i]), i] = torch.fft.ifft(x, norm='ortho', dim=-1).real

    im = torch.zeros((C, M, M))

    k = torch.arange(0, 2 * M - 1)

    for y in range(len(k)):
        z = 0
        for i in range(M):
            for j in range(M):
                if k[y] == i + j:
                    im[:, i, j] = d[:, z, y]
                    z = z + 1

    return im


def IDDCTmode5(D):
    C, M, N = D.shape
    Mh = M // 2
    DD = torch.zeros((C, M, N))

    for i in range(M):
        if i < Mh:
            DD[:, i, :N] = torch.fft.ifft(D[:, i, :N], norm='ortho', dim=-1).real
        else:
            DD[:, i, 1:N - 1] = torch.fft.ifft(D[:, i, :N - 2], norm='ortho', dim=-1).real

    dd = torch.zeros((C, M, N))

    for i in range(N):
        if i == 0:
            dd[:, Mh, 0] = torch.fft.ifft(DD[:, Mh, 0], norm='ortho', dim=-1).real
        elif i == N - 1:
            dd[:, :Mh, N - 1] = torch.fft.ifft(DD[:, :Mh, N - 1], norm='ortho', dim=-1).real
        else:
            dd[:, :M, i] = torch.fft.ifft(DD[:, :M, i], norm='ortho', dim=-1).real

    du = torch.zeros((C, Mh, N - 1))
    dl = torch.zeros((C, Mh, N - 1))

    du[:, :Mh, 0:N - 1] = dd[:, :Mh, 1:N]
    dl[:, :Mh, 0] = dd[:, :Mh, 0]
    dl[:, :Mh, 1:N - 1] = dd[:, Mh:M, 1:N - 1]

    d = torch.zeros((C, M, M))
    d[:, :Mh, :] = du
    d[:, Mh:, :] = dl

    return d
