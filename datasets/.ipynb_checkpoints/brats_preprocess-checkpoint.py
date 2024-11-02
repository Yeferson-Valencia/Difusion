'this file is for transfering 3D BRaTs MRI to 2D Slices of jpg image for training'
import os
import argparse

import numpy as np
import nibabel as nib

def normalize_image(img):

    # Changed to normalize between -1 and 1.
    # 0 to 1 Normalization
    epsilon = 1e-5
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + epsilon)

    # -1 to 1 normalization
    img_normalized = (img_normalized - 0.5) * 2.0
    
    return img_normalized
    
def get_label_file(img_root, img_name):
    masks_dir = os.path.join(img_root, 'test', img_name, 'Masks')
    label_files = os.listdir(masks_dir)

    label_patterns = ['r1_mask', 'r2_mask', 'r1_mask_resampled', 'r2_mask_resampled'] 

    for file in label_files:
        for pattern in label_patterns:
            if pattern in file:
                return os.path.join(masks_dir, file)
    print(f"No se encontró archivo de etiqueta para '{img_name}' con los patrones dados.")
    return None

def nii2np_train(img_root, img_name, upper_per, lower_per, output_root_train=None, modality=None):
    modality = modality.split(',')
    for mod_num in range(len(modality)):
        img_file = os.path.join(img_root, 'train', img_name, img_name + '_' + modality[mod_num] + '.nii.gz')
        img = nib.load(img_file)
        img = img.get_fdata()
        img_original = img
        img = normalize_image(img_original)
        
        img_file_label = os.path.join(img_root, img_name, img_name + '_' + 'seg' + '.nii.gz')
        
        for slice_index in range(img.shape[2]):
            img_slice = img[:, :, slice_index]
            dirs_mod = os.path.join(output_root_train, modality[mod_num])
            if not os.path.exists(dirs_mod):
                os.makedirs(dirs_mod)
            filename = os.path.join(dirs_mod, img_name + '_' + modality[mod_num] + '_' + format(slice_index, '03'))
            np.save(filename, img_slice)

        img_file_label = os.path.join(img_root, 'train', img_name, 'Masks', img_name + '_' + 'r1'+ '_' + 'mask' + '.nii.gz')
        img_label = nib.load(img_file_label)
        img_label = img_label.get_fdata()
        img_label = img_label.astype(np.uint8)

        if mod_num == 0:
            dirs_seg = os.path.join(output_root_train, 'seg')
            if not os.path.exists(dirs_seg):
                os.makedirs(dirs_seg)
            for slice_index in range(img.shape[2]):
                filename_seg = os.path.join(dirs_seg, img_name + '_seg_' + format(slice_index, '03'))
                img_slice_seg = img_label[:, :, slice_index]
                np.save(filename_seg, img_slice_seg)

            dirs_brainmask = os.path.join(output_root_train, 'brainmask')
            if not os.path.exists(dirs_brainmask):
                os.makedirs(dirs_brainmask)
            for slice_index in range(img.shape[2]):
                filename_brainmask = os.path.join(dirs_brainmask, img_name + '_brainmask_' + format(slice_index, '03'))
                img_brainmask = (img_original > 0).astype(int)
                img_slice_brainmask = img_brainmask[:, :, slice_index]
                np.save(filename_brainmask, img_slice_brainmask)


import os
import nibabel as nib
import numpy as np

def nii2np_test(img_root, img_name, upper_per, lower_per, output_root_test=None, modality=None):
    modality = modality.split(',')
    for mod_num in range(len(modality)):
        img_file = os.path.join(img_root, 'test', img_name, img_name + '_' + modality[mod_num] + '.nii.gz')
        img = nib.load(img_file)
        img = img.get_fdata()
        img_original = img
        img = normalize_image(img_original)

        for slice_index in range(img.shape[2]):
            img_slice = img[:, :, slice_index]

            dirs_mod = os.path.join(output_root_test, modality[mod_num])
            if not os.path.exists(dirs_mod):
                os.makedirs(dirs_mod)
            filename = os.path.join(dirs_mod, img_name + '_' + modality[mod_num] + '_' + format(slice_index, '03'))
            np.save(filename, img_slice)

        img_file_label = get_label_file(img_root, img_name)
        
        if img_file_label:
            img_label = nib.load(img_file_label)
            img_label = img_label.get_fdata()
            img_label = img_label.astype(np.uint8)

        if mod_num == 0:
            dirs_seg = os.path.join(output_root_test, 'seg')
            if not os.path.exists(dirs_seg):
                os.makedirs(dirs_seg)
            for slice_index in range(img.shape[2]):
                filename_seg = os.path.join(dirs_seg, img_name + '_seg_' + format(slice_index, '03'))
                img_slice_seg = img_label[:, :, slice_index]
                np.save(filename_seg, img_slice_seg)

            dirs_brainmask = os.path.join(output_root_test, 'brainmask')
            if not os.path.exists(dirs_brainmask):
                os.makedirs(dirs_brainmask)
            for slice_index in range(img.shape[2]):
                filename_brainmask = os.path.join(dirs_brainmask, img_name + '_brainmask_' + format(slice_index, '03'))
                img_brainmask = (img_original > 0).astype(int)
                img_slice_brainmask = img_brainmask[:, :, slice_index]
                np.save(filename_brainmask, img_slice_brainmask)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="the directory in which the data is stored", type=str, default='/home/gustavo_pupils/data/Datasets/BraTS/Complete_DS')
    parser.add_argument("--output_dir", help="the directory to store the preprocessed data", type=str, default='/home/gustavo_pupils/data/Datasets/BraTS/complete_dataset')
    parser.add_argument("--modality", help="The generated modality, like 't1', 't2', or 'flair'. Multi-modality separate by ',' without space, like 't1,t2'", type=str,default='ncct,adc,dwi,flair')
    parser.add_argument("--upper_per", help="The upper percentage of brain area to be normalized, the value needs to be within [0-1], like 0.9", type=float, default=0.9)
    parser.add_argument("--lower_per", help="The lower percentage of brain area to be normalized, the value needs to be within [0-1], like 0.02", type=float, default=0.02)
    args = parser.parse_args()
    img_root = args.data_dir
    img_output_root = args.output_dir
    img_output_root_train = os.path.join(img_output_root, 'train')
    img_output_root_test = os.path.join(img_output_root, 'test')
    train_txt = './apis_split_training.txt'
    test_txt = './apis_split_testing.txt'

    MOD = args.modality
    with open(train_txt) as file:
        for path in file:
            nii2np_train(img_root, path[:-1], upper_per=args.upper_per, lower_per=args.lower_per, output_root_train=img_output_root_train, modality=MOD)
    with open(test_txt) as file:
        for path in file:
            nii2np_test(img_root, path[:-1], upper_per=args.upper_per, lower_per=args.lower_per, output_root_test=img_output_root_test, modality=MOD)