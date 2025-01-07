import csv


def get_meta_info():
    info_file = './datasets/kadid10k/dmos.csv'

    save_meta_path = './datasets/meta_info/meta_info_KADID10kDataset.csv'
    with open(info_file, 'r') as f, open(save_meta_path, 'w+') as sf:
        csvreader = csv.reader(f)
        head = next(csvreader)
        print(head)

        new_head = ['ref_name', 'dist_name', 'dmos', 'std']
        csvwriter = csv.writer(sf)
        csvwriter.writerow(new_head)
        for idx, row in enumerate(csvreader):
            print(row)
            dis_name = row[0]
            ref_name = row[1]
            dmos = row[2]
            std = row[3]
            csvwriter.writerow([ref_name, dis_name, dmos, std])


if __name__ == '__main__':
    get_meta_info()
