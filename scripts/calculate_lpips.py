import cv2
import glob
import numpy as np
import torch
import os.path as osp
from torchvision.transforms.functional import normalize

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def main():
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = 'datasets/of/tcl/test/target'
    folder_restored = 'results/Test_DTNet_tclv2.yml/visualization/MultiFrameVD'
    # crop_border = 4
    suffix = ''
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + '.png'), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.
        # img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + '.jpg'), cv2.IMREAD_UNCHANGED).astype(
        #     np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
        lpips_val_item = lpips_val.item()

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val_item:.6f}.')
        lpips_all.append(lpips_val_item)

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':

    main()
