import csv
import os


def write_csv(data=None, csv_path=None, save_name='result'):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

    with open(csv_path + "/" + save_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(len(data)):
            name = data[i][0]
            predict = data[i][1]
            w.writerow([name, predict])
    f.close()
