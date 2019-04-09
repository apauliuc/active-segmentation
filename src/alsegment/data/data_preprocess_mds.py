import pickle
import shutil

import numpy as np
import os

from PIL import Image
import SimpleITK as SiTK


def mds_preprocess_scans(root_dir, max_clip=100, clip_max_to_0=False):
    scans_name = 'arr_scan_no_skull.npy'

    if not clip_max_to_0:
        clipped_scan_name = f'arr_scan_clipped_{max_clip}'
        mha_scan_name = f'scan_clipped_{max_clip}'
    else:
        clipped_scan_name = f'arr_scan_clipped_{max_clip}_to_0'
        mha_scan_name = f'scan_clipped_{max_clip}_to_0'

    for acc_nr in os.listdir(root_dir):
        scan_path = os.path.join(root_dir, acc_nr)

        if os.path.isdir(scan_path):
            scan = np.load(os.path.join(scan_path, scans_name))

            if not clip_max_to_0:
                scan_clipped = np.clip(scan, a_min=0, a_max=max_clip)
            else:
                scan_clipped = np.clip(scan, a_min=0, a_max=None)
                scan_clipped[scan_clipped >= max_clip] = 0

            np.save(os.path.join(scan_path, f'{clipped_scan_name}.npy'), scan_clipped)

            SiTK.WriteImage(SiTK.GetImageFromArray(scan_clipped), os.path.join(scan_path, f'{mha_scan_name}.mha'))

    return f'{clipped_scan_name}.npy'


def mds_process_scans_from_list(root_dir, save_path, scan_name, scans_to_separate):
    count = 0
    file_list = []

    loaded_scans = []

    for acc_nr in scans_to_separate:
        acc_dir = os.path.join(root_dir, acc_nr)
        if os.path.isdir(acc_dir):
            scan = np.load(os.path.join(acc_dir, scan_name))
            segment = np.load(os.path.join(acc_dir, 'arr_segment.npy'))

            loaded_scans.extend(np.split(scan, scan.shape[0]))

            for i in np.arange(scan.shape[0]):
                in_name = '%s_%i' % (acc_nr, i)
                out_name = '%s_%i_seg' % (acc_nr, i)

                file_list.append(in_name)

                np.save(os.path.join(save_path, in_name), scan[i, :, :])
                np.save(os.path.join(save_path, out_name), segment[i, :, :])

                count += 1

    with open(os.path.join(save_path, 'file_list.pkl'), 'wb') as f:
        pickle.dump(file_list, f)

    return count, loaded_scans


def mds_prepare_prediction_dir(root_dir, save_path, npy_scan_name, predict_list):
    old_names = [npy_scan_name, f'{npy_scan_name[4:-4]}.mha', 'segmentation.mha']
    for acc_nr in predict_list:
        acc_dir = os.path.join(root_dir, acc_nr)
        prediction_save_dir = os.path.join(save_path, acc_nr)
        os.makedirs(prediction_save_dir, exist_ok=True)
        if os.path.isdir(acc_dir):
            new_names = [f'{acc_nr}_scan.npy', f'{acc_nr}.mha', f'{acc_nr}_segmentation.mha']
            for old, new in zip(old_names, new_names):
                new_path = os.path.join(os.path.join(prediction_save_dir, new))
                if os.path.exists(new_path):
                    os.remove(new_path)
                shutil.copy(os.path.join(acc_dir, old), prediction_save_dir)
                os.rename(os.path.join(prediction_save_dir, old), os.path.join(prediction_save_dir, new))


def mds_separate_scans_to_slices(root_dir, save_path, scan_name, dummy_dataset=False):
    # Delete and recreate folders
    train_path = os.path.join(save_path, 'train')
    val_path = os.path.join(save_path, 'val')
    predict_path = os.path.join(save_path, 'predict')

    if os.path.isdir(train_path):
        shutil.rmtree(train_path)
    if os.path.isdir(val_path):
        shutil.rmtree(val_path)

    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(predict_path, exist_ok=True)

    # Preparation
    all_patients = os.listdir(root_dir)
    val_patients = ['508002', '397963', '1451713']
    train_patients = [x for x in all_patients if x not in val_patients]

    if dummy_dataset:
        train_patients = ['1025819']
        val_patients = train_patients

    train_count, train_scans = mds_process_scans_from_list(root_dir, train_path, scan_name, train_patients)
    val_count, val_scans = mds_process_scans_from_list(root_dir, val_path, scan_name, val_patients)
    mds_prepare_prediction_dir(root_dir, predict_path, scan_name, val_patients)

    print("Saved %i images to train" % train_count)
    print("Saved %i images to val" % val_count)
    print("Saved %i total images" % (train_count + val_count))

    scans_list = train_scans + val_scans

    dataset_scan = np.stack(scans_list)
    norm = {
        'mean': dataset_scan.mean(),
        'std': dataset_scan.std()
    }

    with open(os.path.join(save_path, 'norm_data.pkl'), 'wb') as f:
        pickle.dump(norm, f)
