import os
import sys
import argparse
import random
import cv2
import numpy as np
import torch as th
import blobfile as bf
from pathlib import Path
from skimage.transform import resize
import matplotlib.pyplot as plt
from datasets import loader
from configs import get_config
from utils import logger
from utils.script_util import create_gaussian_diffusion, create_score_model
from utils.binary_metrics import assd_metric, sensitivity_metric, precision_metric
sys.path.append(str(Path.cwd()))

def normalize(img, _min=None, _max=None):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def patient_data_exists(patient_id, data_dir, input_type):
    patient_data_dir = os.path.join(data_dir, 'test', input_type)
    print(f"Checking data in directory: {patient_data_dir} for patient: {patient_id}")
    
    if not os.path.exists(patient_data_dir):
        print(f"Directory does not exist: {patient_data_dir}")
        return False
    
    all_files = os.listdir(patient_data_dir)
    
    for fname in all_files:
        if fname.startswith(patient_id) and fname.endswith('.npy'):
            print(f"Found file for patient {patient_id}: {fname}")
            return True
    return False

def load_patient_data(patient_id, data_dir, input_type, target_size=(256, 256)):
    patient_data_dir = os.path.join(data_dir, 'test', input_type)
    patient_files = sorted([f for f in os.listdir(patient_data_dir) if f.startswith(patient_id) and f.endswith('.npy')])
    patient_data = [np.load(os.path.join(patient_data_dir, f)) for f in patient_files]
    

    resized_data = []
    for img in patient_data:
        if img.shape != target_size:
            img = resize(img, target_size, anti_aliasing=True)
        resized_data.append(img)
    
    resized_data = np.array(resized_data)
    
    resized_data = resized_data[:, np.newaxis, :, :]
    print(f"Shape for {patient_id}: {resized_data.shape}")
    return resized_data, patient_files

def process_patient(patient_id, args, config, diffusion, model_forward):
    logger.log("------------------------------------")
    logger.log(f"Processing patient {patient_id}...")

    if not patient_data_exists(patient_id, args.data_dir, args.input):
        logger.log(f"No data found for patient {patient_id}. Skipping...")
        return

    patient_data, patient_files = load_patient_data(patient_id, args.data_dir, args.input)
    
    if patient_data.shape[1] != config.score_model.num_input_channels:
        raise ValueError(f"Expected {config.score_model.num_input_channels} channels, but got {patient_data.shape[1]} channels")

    patient_data = th.tensor(patient_data).float().cuda()

    # Crear la segmentación sintética para este ejemplo
    patient_seg = th.ones_like(patient_data).cuda()

    model_kwargs = {}

    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model_forward, patient_data, 0,
        (patient_seg.shape[0], config.score_model.num_input_channels, config.score_model.image_size,
         config.score_model.image_size),
        model_name=args.model_name,
        clip_denoised=config.sampling.clip_denoised,
        model_kwargs=model_kwargs,
        eta=config.sampling.eta,
        model_forward_name=args.experiment_name_forward,
        ddim=args.use_ddim
    )

    output_png_folder = "/home/gustavo_pupils/data/Datasets/PPMI/Predictions/output_images/"
    output_npy = "/home/gustavo_pupils/data/Datasets/PPMI/Predictions/output_npy/"
    epsilon = 1e-5

    for i, image_array in enumerate(sample.cpu().numpy()):
        input_image_name = os.path.splitext(patient_files[i])[0] 
        output_path_npy = f"{output_npy}{input_image_name}_pred"
        np.save(output_path_npy, image_array)
        
        
        image_array = image_array[0, :, :] 
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + epsilon)
        image_array = (image_array * 255).astype(np.uint8)
        
        output_path = f"{output_png_folder}{input_image_name}_pred.png"
        cv2.imwrite(output_path, image_array)
    
    return True

def main(args):
    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)
    config = get_config.file_from_dataset(args.dataset)
    if args.experiment_name_forward != 'None':
        experiment_name = args.experiment_name_forward
    else:
        raise Exception("Experiment name does not exist")
    logger.configure(Path(experiment_name) / "score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    if args.model_name == 'unet':
        image_level_cond_forward = False
        image_level_cond_backward = False
    elif args.model_name == 'diffusion':
        image_level_cond_forward = True
        image_level_cond_backward = False
    else:
        raise Exception("Model name does not exist")
    diffusion = create_gaussian_diffusion(config, args.timestep_respacing)
    model_forward = create_score_model(config, image_level_cond_forward)

    filename = "model020000.pt"
    with bf.BlobFile(bf.join(logger.get_dir(), filename), "rb") as f:
        model_forward.load_state_dict(
            th.load(f.name, map_location=th.device('cuda'))
        )
    model_forward.to(th.device('cuda'))

    if config.score_model.use_fp16:
        model_forward.convert_to_fp16()

    model_forward.eval()

    logger.log("starting patient processing...")

    processed_patients_file = "processed_patients.txt"
    if os.path.exists(processed_patients_file):
        with open(processed_patients_file, 'r') as f:
            processed_patients = set(f.read().splitlines())
    else:
        processed_patients = set()

    patient_ids = [f"test_{i:03d}" for i in range(29)]
    for patient_id in patient_ids:
        if patient_id in processed_patients:
            logger.log(f"Patient {patient_id} already processed. Skipping...")
            continue
        success = process_patient(patient_id, args, config, diffusion, model_forward)
        if success:
            with open(processed_patients_file, 'a') as f:
                f.write(f"{patient_id}\n")
            logger.log(f"Finished processing patient {patient_id}")

def reseed_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    parser.add_argument("--dataset", help="brats", type=str, default='brats')
    parser.add_argument("--input", help="input modality, choose from flair, t2, t1", type=str, default='mri')
    parser.add_argument("--data_dir", help="data directory", type=str, default='/home/gustavo_pupils/data/Datasets/PPMI')
    parser.add_argument("--experiment_name_forward", help="forward model saving file name", type=str, default='diffusion_brats_mri_spect')
    parser.add_argument("--model_name", help="translated model: unet or diffusion", type=str, default='diffusion')
    parser.add_argument("--use_ddim", help="if you want to use ddim during sampling, True or False", type=str, default='True')
    parser.add_argument("--timestep_respacing", help="If you want to rescale timestep during sampling. enter the timestep you want to rescale the diffusion prcess to. If you do not wish to resale the timestep, leave it blank or put 1000.", type=int,
                        default=100)

    args = parser.parse_args()
    print(args.dataset)
    main(args)
