import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_, DCNv2Pack, ResidualBlockNoBN, make_layer
from basicsr.archs.edvr_arch import PCDAlignment

from vd.archs.DDCT_transform_gpu_arch import DDCT_transform
from vd.archs.IDDCT_gpu_arch import IDDCT


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.PReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size,
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        R = full_res_input * coeff[:, 0:3, :, :] + coeff[:, 3:6, :, :]
        G = full_res_input * coeff[:, 6:9, :, :] + coeff[:, 9:12, :, :]
        B = full_res_input * coeff[:, 12:15, :, :] + coeff[:, 15:18, :, :]
        result = torch.cat([R, G, B], dim=1)

        return result


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.is_bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        _, _, h, w = x1.shape
        if self.is_bilinear:
            x1 = F.interpolate(x1, size=(h * 2, w * 2), mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(72, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv2 = ConvBlock(16, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv3 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)

        return output


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.pre = nn.Conv2d(16, 3, 3, 1, 1)
        self.re = nn.Sigmoid()

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.re(self.pre(x))

        return x


class dct_weight(nn.Module):
    def __init__(self, in_c=64, out_c = 64*4):
        super(dct_weight, self).__init__()
        self.out_channels = int(in_c / 2)
        self.conv1 = nn.Conv2d(72, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, out_c, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class directional_dct_layer(nn.Module):
    def __init__(self, in_c=72, out_c=72 * 72):
        super(directional_dct_layer, self).__init__()
        self.ddct_conv = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.GELU()
        self.conv_dct = nn.Conv2d(in_c, out_c, kernel_size=3, stride=8, padding=2, dilation=2, groups=in_c)

    def forward(self, x):
        out = self.act(self.ddct_conv(x)) + x
        out = self.conv_dct(out)
        return out


def check_image_size(h, w, bs):
    new_h = h
    new_w = w
    if h % bs != 0:
        new_h = h + (bs - h % bs)
    if w % bs != 0:
        new_w = w + (bs - w % bs)
    return new_h, new_w


def directional_inverse_dct_layer(img, bs=8, mode=0):
    b, ch, h, w = img.shape
    imt = img.view(b*ch, h, w)
    c, m, n = imt.shape
    new_m, new_n = check_image_size(m, n, bs)
    new_imt = torch.zeros((c, new_m, new_n)).cuda()
    new_imt[:, :m, :n] = imt
    imf = torch.zeros((c, new_m, new_n)).cuda()
    for ii in range(0, new_m, bs):
        for jj in range(0, new_n, bs):
            cb = new_imt[:, ii:ii + bs, jj:jj + bs]
            CB = DDCT_transform(cb, mode)
            cbf = IDDCT(CB, mode)
            imf[:, ii:ii + bs, jj:jj + bs] = cbf
    imf = imf[:, :m, :n]
    didct = imf.view(b, ch, h, w)
    return didct


class MTRB(nn.Module):
    def __init__(self, num_feat=64, block_size=8):
        super(MTRB, self).__init__()
        self.conv = nn.Conv2d(72, 64, 3, 1, 1)
        self.conv_dct_0 = directional_dct_layer(in_c=num_feat, out_c=num_feat)
        self.conv_dct_1 = directional_dct_layer(in_c=num_feat, out_c=num_feat)
        self.conv_dct_3 = directional_dct_layer(in_c=num_feat, out_c=num_feat)
        self.conv_dct_4 = directional_dct_layer(in_c=num_feat, out_c=num_feat)
        self.conv_dct_5 = directional_dct_layer(in_c=num_feat, out_c=num_feat)
        self.conv_dct_6 = directional_dct_layer(in_c=num_feat, out_c=num_feat)
        self.conv_dct_7 = directional_dct_layer(in_c=num_feat, out_c=num_feat)
        self.conv_dct_8 = directional_dct_layer(in_c=num_feat, out_c=num_feat)

        self.dct_weight = dct_weight(in_c=num_feat, out_c=num_feat*8)
        self.in_c = num_feat
        self.bs = block_size

        self.after_rdct = nn.Conv2d(num_feat * 8, 72, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        _, _, h, w = x.size()
        dct_feat = self.act(self.conv(x))  # torch.Size([4, 3, 3, 720, 1280])
        # mode 0
        dct_feat_0 = self.conv_dct_0(dct_feat)
        out_0 = directional_inverse_dct_layer(dct_feat_0, bs=self.bs, mode=0)
        out_0 = F.interpolate(out_0, size=(h, w), mode='bilinear', align_corners=False)
        # mode 1
        dct_feat_1 = self.conv_dct_1(dct_feat)
        out_1 = directional_inverse_dct_layer(dct_feat_1, bs=self.bs, mode=1)
        out_1 = F.interpolate(out_1, size=(h, w), mode='bilinear', align_corners=False)
        # mode 3
        dct_feat_3 = self.conv_dct_3(dct_feat)
        out_3 = directional_inverse_dct_layer(dct_feat_3, bs=self.bs, mode=3)
        out_3 = F.interpolate(out_3, size=(h, w), mode='bilinear', align_corners=False)
        # mode 4
        dct_feat_4 = self.conv_dct_4(dct_feat)
        out_4 = directional_inverse_dct_layer(dct_feat_4, bs=self.bs, mode=4)
        out_4 = F.interpolate(out_4, size=(h, w), mode='bilinear', align_corners=False)
        # mode 5
        dct_feat_5 = self.conv_dct_5(dct_feat)
        out_5 = directional_inverse_dct_layer(dct_feat_5, bs=self.bs, mode=5)
        out_5 = F.interpolate(out_5, size=(h, w), mode='bilinear', align_corners=False)
        # mode 6
        dct_feat_6 = self.conv_dct_4(dct_feat)
        out_6 = directional_inverse_dct_layer(dct_feat_6, bs=self.bs, mode=6)
        out_6 = F.interpolate(out_6, size=(h, w), mode='bilinear', align_corners=False)
        # mode 7
        dct_feat_7 = self.conv_dct_7(dct_feat)
        out_7 = directional_inverse_dct_layer(dct_feat_7, bs=self.bs, mode=7)
        out_7 = F.interpolate(out_7, size=(h, w), mode='bilinear', align_corners=False)
        # mode 8
        dct_feat_8 = self.conv_dct_8(dct_feat)
        out_8 = directional_inverse_dct_layer(dct_feat_8, bs=self.bs, mode=8)
        out_8 = F.interpolate(out_8, size=(h, w), mode='bilinear', align_corners=False)


        dct_weight = self.dct_weight(x)
        out_0 = torch.mul(out_0, dct_weight[:, 0: self.in_c, :, :])
        out_1 = torch.mul(out_1, dct_weight[:, self.in_c:2 * self.in_c, :, :])
        out_3 = torch.mul(out_3, dct_weight[:, 2 * self.in_c:3 * self.in_c, :, :])
        out_4 = torch.mul(out_4, dct_weight[:, 3 * self.in_c:4 * self.in_c, :, :])
        out_5 = torch.mul(out_5, dct_weight[:, 4 * self.in_c:5 * self.in_c, :, :])
        out_6 = torch.mul(out_6, dct_weight[:, 5 * self.in_c:6 * self.in_c, :, :])
        out_7 = torch.mul(out_7, dct_weight[:, 6 * self.in_c:7 * self.in_c, :, :])
        out_8 = torch.mul(out_8, dct_weight[:, 7 * self.in_c:, :, :])

        out = torch.cat([out_0, out_1, out_3, out_4, out_5, out_6, out_7, out_8], dim=1)

        out = self.after_rdct(out)
        return out


@ARCH_REGISTRY.register()
class DTNet_g(nn.Module):
    def __init__(self, nf=72):
        super(DTNet_g, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 4, 4)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=nf)
        self.conv_l2_1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.mtrb = MTRB(num_feat=64, block_size=8)
        self.pcd_align = PCDAlignment(num_feat=nf, deformable_groups=8)

        self.before_fusion = nn.Conv2d(nf * 3, nf, 3, 1, 1)

        self.conv_before_upsample1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample1 = nn.Sequential(nn.Conv2d(nf, nf * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4))
        self.conv_last1 = nn.Conv2d(nf, 3, 3, 1, 1)

        self.apply(self._init_weights)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        b, t, c, h, w = x.size()  # torch.Size([4, 3, 3, 720, 1280])

        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.mtrb(feat_l1) + feat_l1    # MTRB
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h // 4, w // 4)
        feat_l2 = feat_l2.view(b, t, -1, h // 8, w // 8)
        feat_l3 = feat_l3.view(b, t, -1, h // 16, w // 16)

        # PCD alignment
        ref_feat_l = [feat_l1[:, 1, :, :, :].clone(), feat_l2[:, 1, :, :, :].clone(), feat_l3[:, 1, :, :, :].clone()]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h // 4, w // 4)
        x = self.lrelu(self.before_fusion(aligned_feat))
        out_i1 = self.conv_last1(self.upsample1(self.conv_before_upsample1(x)))
        return out_i1, aligned_feat


@ARCH_REGISTRY.register()
class DTNet_f(nn.Module):
    def __init__(self, nf=72):
        super(DTNet_f, self).__init__()
        self.upsample_i0 = nn.Sequential(nn.Conv2d(nf, nf * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4),
                                         nn.LeakyReLU(inplace=True))
        self.upsample_i1 = nn.Sequential(nn.Conv2d(nf, nf * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4),
                                         nn.LeakyReLU(inplace=True))
        self.upsample_i2 = nn.Sequential(nn.Conv2d(nf, nf * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4),
                                         nn.LeakyReLU(inplace=True))
        self.guide_i0 = GuideNN()
        self.guide_i1 = GuideNN()
        self.guide_i2 = GuideNN()

        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        self.u_net = UNet(n_channels=3)
        # self.smooth = nn.PReLU()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=27, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )

        self.x_r_fusion = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.downsample = nn.AdaptiveAvgPool2d((96, 256))
        self.p = nn.PReLU()

        self.r_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.g_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.b_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, out_i1, aligned_feat):
        x_r = F.interpolate(out_i1, size=(256, 256), mode='bicubic', align_corners=True)
        # b * 3 * 64 * 256 -> b * 12 * 16 * 16 * 16
        coeff = self.downsample(self.u_net(x_r)).reshape(-1, 18, 16, 16, 16)

        aligned_feat_i0, aligned_feat_i1, aligned_feat_i2 = torch.chunk(aligned_feat, 3, dim=1)

        guidance_i0 = self.guide_i0(self.upsample_i0(aligned_feat_i0))
        guidance_i1 = self.guide_i1(self.upsample_i1(aligned_feat_i1))
        guidance_i2 = self.guide_i2(self.upsample_i2(aligned_feat_i2))

        slice_coeffs_i0 = self.slice(coeff, guidance_i0)
        slice_coeffs_i1 = self.slice(coeff, guidance_i1)
        slice_coeffs_i2 = self.slice(coeff, guidance_i2)
        output_i0 = self.apply_coeffs(slice_coeffs_i0, self.p(self.r_point(out_i1)))
        output_i1 = self.apply_coeffs(slice_coeffs_i1, self.p(self.g_point(out_i1)))
        output_i2 = self.apply_coeffs(slice_coeffs_i2, self.p(self.b_point(out_i1)))

        output = torch.cat((output_i0, output_i1, output_i2), dim=1)
        output = self.fusion(output)
        output = self.p(self.x_r_fusion(output) * out_i1 - output + 1)
        return output


if __name__ == '__main__':
    x = torch.rand(1, 3, 1088, 1920)
    model_g = DTNet_g()
    model_f = DTNet_f()
    out1, x_feat = model_g(x)
    out = model_f(out1, x_feat)
    print(out1.shape)
    print(x_feat.shape)
    print(out.shape)
