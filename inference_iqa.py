import argparse
import glob
import os
from PIL import Image
from pyiqa.models.inference_model import InferenceModel


def main():
    """Inference demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, help='input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')
    parser.add_argument(
        '-m',
        '--metric_mode',
        type=str,
        default='FR',
        help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument('-n', '--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    parser.add_argument('--model_path', type=str, default=None, help='Weight path for CNN based models.')
    parser.add_argument('--img_range', type=float, default=1.0, help='Max value of image tensor.')
    parser.add_argument(
        '--input_size', type=int, nargs='+', default=None, help='size of input image. (H, W) for tuple input.')
    parser.add_argument(
        '--mean', type=float, nargs='+', default=None, metavar='MEAN', help='Override mean pixel value of dataset')
    parser.add_argument(
        '--std', type=float, nargs='+', default=None, metavar='STD', help='Override std deviation of of dataset')
    parser.add_argument('--save_file', type=str, default=None, help='path to save results.')

    args = parser.parse_args()

    metric_name = args.metric_name.lower()

    # set up IQA model
    iqa_model = InferenceModel(metric_name, args.metric_mode, args.model_path, args.img_range, args.input_size,
                               args.mean, args.std)
    metric_mode = iqa_model.metric_mode

    if os.path.isfile(args.input):
        input_paths = [args.input]
        if args.ref is not None:
            ref_paths = [args.ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.input, '*')))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))

    if args.save_file:
        sf = open(args.save_file, 'w')

    avg_score = 0
    test_img_num = len(input_paths)
    for idx, img_path in enumerate(input_paths):
        img_name = os.path.basename(img_path)
        tar_img = Image.open(img_path)
        if metric_mode == 'FR':
            ref_img_path = ref_paths[idx]
            ref_img = Image.open(ref_img_path)
        else:
            ref_img = None
        score = iqa_model.test(tar_img, ref_img)
        avg_score += score
        print(f'{metric_name} score of {img_name} is: {score}')
        if args.save_file:
            sf.write(f'{img_name}\t{score}\n')
    avg_score /= test_img_num
    if test_img_num > 1:
        print(f'Average {metric_name} score of {args.input} with {test_img_num} images is: {avg_score}')
    if args.save_file:
        sf.close()

    if args.save_file:
        print(f'Done! Results are in {args.save_file}.')
    else:
        print(f'Done!')


if __name__ == '__main__':
    main()
