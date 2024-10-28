import csv
import json
from tqdm import tqdm


def create_cfgiqa_meta_info():
    data_root = '../datasets/CGFIQA/'

    json_files = [
        '../datasets/CGFIQA/test/label.json',
        '../datasets/CGFIQA/train/label.json',
        '../datasets/CGFIQA/val/label.json',
    ]

    save_meta_file = '../datasets/meta_info/meta_info_CGFIQADataset.csv'
    file = open(save_meta_file, 'w')
    writer = csv.writer(file)
    writer.writerow(['img_path', 'score', 'official_split'])

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            img_list = data['Image']
            labels = data['MOS']

            for img, label in zip(img_list, labels):
                if 'train' in json_file:
                    img_path = f'train/img/{img}'
                    split = 'train'
                elif 'test' in json_file:
                    img_path = f'test/img/{img}'
                    split = 'test'
                else:
                    img_path = f'val/img/{img}'
                    split = 'val'

                writer.writerow([img_path, label, split])

            # for key, value in data.items():
            #     img_path = data_root + key
            #     print(img_path, label, score)
            #     exit()
    file.close()

if __name__ == '__main__':
    create_cfgiqa_meta_info()

    