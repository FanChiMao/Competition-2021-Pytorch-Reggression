import csv
import numpy as np
import os


def write_csv(data, csv_path=None, save_name='result'):

    if not os.path.exists(csv_path):
        os.mkdir(csv_path)


    with open(csv_path + "/" + save_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(len(data)):
            box = data[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            boundary_height = y2 - y1
            boundary_width = x2 - x1
            area = boundary_height * boundary_width
            center_x = round((x1 + x2) * 0.5)
            center_y = round((y1 + y2) * 0.5)

            w.writerow([center_x, center_y])

    f.close()

    return count

