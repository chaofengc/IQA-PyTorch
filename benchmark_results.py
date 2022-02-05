import argparse
import yaml
import csv
from pyiqa.models.inference_model import InferenceModel
from pyiqa.data import build_dataset, build_dataloader
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.options import ordered_yaml
from pyiqa.metrics import calculate_plcc, calculate_srcc, calculate_krcc
from tqdm import tqdm

def main():
    """benchmark test demo for pyiqa. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('-num_gpu', type=int, default=0, help='use gpu or not')
    args = parser.parse_args()

    # parse yaml options
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    
    save_result_path = opt['save_result_path']

    csv_file = open(save_result_path, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Metric name'] + [ds['name']+'(PLCC/SRCC/KRCC)' for ds in opt['datasets'].values()]) 

    for metric_config in opt['metrics'].values():
        metric_name = metric_config['name']
        # if metric_name exist in default config, load default config first  
        if metric_name in DEFAULT_CONFIGS.keys():
            metric_opts = DEFAULT_CONFIGS[metric_name]['metric_opts']
            if 'metric_opts' in metric_config.keys():
                metric_opts.update(metric_config['metric_opts'])
        else:
            metric_opts = metric_config['metric_opts']
        metric_mode = metric_config['metric_mode']
        iqa_model = InferenceModel(metric_mode, metric_opts)

        results_row = [metric_name]
        for ds in opt['datasets'].values():
            ds['phase'] = 'test'
            dataset = build_dataset(ds)
            dataset_name = ds['name']
            dataloader = build_dataloader(dataset, ds, num_gpu=args.num_gpu)
            gt_labels = []
            result_scores = []
            pbar = tqdm(total=len(dataloader), unit='image')
            for data in dataloader:
                gt_labels.append(data['mos_label'].cpu().item())
                if metric_mode == 'FR':
                    tar_img = data['img'] 
                    ref_img = data['ref_img'] 
                    iqa_score = iqa_model.test(tar_img, ref_img)
                else:
                    tar_img = data['img'] 
                    iqa_score = iqa_model.test(tar_img)
                result_scores.append(iqa_score)
                pbar.update(1)
                pbar.set_description(f'Test {metric_name} on {dataset_name}')
            pbar.close()
            
            plcc_score = round(calculate_plcc(result_scores, gt_labels), 4)
            srcc_score = round(calculate_srcc(result_scores, gt_labels), 4)
            krcc_score = round(calculate_krcc(result_scores, gt_labels), 4)
            results_row.append(f'{plcc_score}/{srcc_score}/{krcc_score}')
            print(f'Results of metric {metric_name} on {dataset_name} is [PLCC|SRCC|KRCC]: {plcc_score}, {srcc_score}, {krcc_score}')
        csv_writer.writerow(results_row)
    csv_file.close()
        
if __name__ == '__main__':
    main()

