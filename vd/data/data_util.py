import os
import os.path as osp
import glob
import cv2
import numpy as np

from basicsr.utils import scandir



def multiframe_paired_paths_from_folders_train(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_folder, gt_folder = folders

    gt_paths = list(scandir(gt_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('.jpg')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_path = osp.join(gt_folder, gt_name + '.jpg')
        scene_idx = gt_name.split('_')[0]
        patch_idx = gt_name[-4:]

        lq_1_idx = int(gt_name.split('_')[1])  # 42
        if lq_1_idx != 0 and lq_1_idx != 59:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
        elif lq_1_idx == 0:
            lq_0_name = gt_name  # 0
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)  # 1
        else:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)  # 58
            lq_2_name = gt_name  # 59

        lq_0_path = osp.join(input_folder, lq_0_name + '.jpg')
        lq_1_path = osp.join(input_folder, gt_name + '.jpg')
        lq_2_path = osp.join(input_folder, lq_2_name + '.jpg')
        paths.append(dict(
            [('lq_0_path', lq_0_path), ('lq_1_path', lq_1_path), ('lq_2_path', lq_2_path), ('gt_path', gt_path),
             ('key', gt_name)]))
    return paths


def multiframe_paired_paths_from_folders_val(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_folder, gt_folder = folders

    gt_paths = list(scandir(gt_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('.jpg')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_path = osp.join(gt_folder, gt_name + '.jpg')
        scene_idx = gt_name.split('_')[0]

        lq_1_idx = int(gt_name.split('_')[1])  # 42
        if lq_1_idx != 0 and lq_1_idx != 59:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
        elif lq_1_idx == 0:
            lq_0_name = gt_name  # 0
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)  # 1
        else:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5) # 58
            lq_2_name = gt_name  # 59

        lq_0_path = osp.join(input_folder, lq_0_name + '.jpg')
        lq_1_path = osp.join(input_folder, gt_name + '.jpg')
        lq_2_path = osp.join(input_folder, lq_2_name + '.jpg')
        paths.append(dict(
            [('lq_0_path', lq_0_path), ('lq_1_path', lq_1_path), ('lq_2_path', lq_2_path), ('gt_path', gt_path),
             ('key', gt_name)]))
    return paths



def tensor2numpy(tensor):
    img_np = tensor.squeeze().numpy()
    img_np[img_np < 0] = 0
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    return img_np.astype(np.float32)


def imwrite_gt(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)

    img = img.clip(0, 1.0)
    uint8_image = np.round(img * 255.0).astype(np.uint8)
    cv2.imwrite(img_path, uint8_image)
    return None


def read_img(img_path):
    img = cv2.imread(img_path, -1)
    return img / 255.
