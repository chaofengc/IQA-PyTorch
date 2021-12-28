import argparse
import cv2
import glob
import numpy as np
import os
import math
import torch
import torchvision as tv
from PIL import Image
from pyiqa.utils import imwrite
from pyiqa.models.inference_model import InferenceModel
from pyiqa.default_model_configs import DEFAULT_CONFIGS

def main():
    """Inference demo for pyiqa. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mode', type=str, default='dir', help='Input image or dir.')
    parser.add_argument('--img_path', type=str, default=None, help='input image path.')
    parser.add_argument('--ref_img_path', type=str, default=None, help='reference image path if needed.')
    parser.add_argument('--input_dir', type=str, default=None, help='input dir path.')
    parser.add_argument('--metric_mode', type=str, default='FR', help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument('--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    parser.add_argument('--ref_dir', type=str, default=None, help='reference dir path if needed.')
    parser.add_argument('--model_path', type=str, default=None, help='Weight path for CNN based models.')
    parser.add_argument('--img_range', type=float, default=1.0, help='Max value of image tensor.')
    parser.add_argument('--input_size', type=int, nargs='+', default=None, help='size of input image. (H, W) for tuple input.')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                                help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                                help='Override std deviation of of dataset')
    parser.add_argument('--save_file', type=str, default=None, help='path to save results.')

    args = parser.parse_args()

    metric_name = args.metric_name

    # set up IQA model 
    if metric_name in DEFAULT_CONFIGS.keys():
        iqa_model = InferenceModel(metric_name, **DEFAULT_CONFIGS[metric_name])
        metric_mode = DEFAULT_CONFIGS[metric_name]['metric_mode']
    else:
        iqa_model = InferenceModel(metric_name, args.metric_mode, args.model_path, 
                args.img_range, args.input_size, args.mean, args.std)
        metric_mode = args.metric_mode

    if args.input_mode == 'image':
        tar_img_path = args.img_path
        img_name = os.path.basename(tar_img_path) 
        print(f'Calculating {metric_name} score of {img_name} ...')
        tar_img = Image.open(tar_img_path)
        if metric_mode == 'FR':
            ref_img_path = args.ref_img_path
            ref_img = Image.open(ref_img_path)
        else:
            ref_img = None
        score = iqa_model.test(tar_img, ref_img)
        print(f'{metric_name} score of {img_name} is: {score}')
        if args.save_file:
            with open(args.save_file, 'w') as sf:
                sf.write(f'{img_name}\t{score}\n')
            
    elif args.input_mode == 'dir':
        img_list = sorted(glob.glob(os.path.join(args.input_dir, '*')))
        if args.save_file:
            sf = open(args.save_file, 'w')
        avg_score = 0
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            tar_img = Image.open(img_path) 
            if metric_mode == 'FR':
                ref_img_path = os.path.join(args.ref_dir, img_name)
                ref_img = Image.open(ref_img_path)
            else:
                ref_img = None
            score = iqa_model.test(tar_img, ref_img)
            avg_score += score
            print(f'{metric_name} score of {img_name} is: {score}')
            if args.save_file:
                sf.write(f'{img_name}\t{score}\n')
        avg_score /= len(img_list)
        print(f'Average {metric_name} score of {args.input_dir} with {len(img_list)} images is: {avg_score}')
        if args.save_file:
            sf.close()
  
    if args.save_file:
        print(f'Done! Results are in {args.save_file}.')
    else:
        print(f'Done!')


if __name__ == '__main__':
    main()

