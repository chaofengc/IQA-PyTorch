import csv
from glob import glob


def get_meta_info():
    train_label_folder = './datasets/PIPAL/Train_Label/'

    name_labels = []
    for f in sorted(glob(train_label_folder + '*.txt')):
        name_labels += [x.strip().split(',') for x in open(f).readlines()]

    new_head = ['hq_name', 'dist_name', 'elo_score']

    save_meta_path = './datasets/meta_info/meta_info_PIPALDataset.csv'
    with open(save_meta_path, 'w') as sf:
        csvwriter = csv.writer(sf)
        csvwriter.writerow(new_head)
        for n, l in name_labels:
            dist_name = n
            elo_score = l
            hq_name = dist_name.split('_')[0] + '.bmp'
            csvwriter.writerow([hq_name, dist_name, elo_score])


if __name__ == '__main__':
    get_meta_info()
