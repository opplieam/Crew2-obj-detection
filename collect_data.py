import time
import os
import csv
import cv2
import numpy as np

from datetime import datetime
from getkeys import key_check
from grabscreen import grab_screen

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return sizeof_fmt(total_size)


def keys_to_output(keys):
    """
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    """

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def count_down():
    for i in list(range(4))[::-1]:
        print("Script start in ", i + 1)
        time.sleep(1)


def main():
    count_down()
    binary_image_name = 'D:/data_m2n/{}.npy'
    csv_file_name = 'D:/data_m2n.csv'
    paused = True
    print('STARTING!!!')
    with open(csv_file_name, mode="a") as my_file:
        writer = csv.writer(
            my_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        # writer.writerow(
        #     ["file_name", "w", "s", "a", "d", "wa", "wd", "sa", "sd", 'nk']
        # )

        while True:
            keys = key_check()
            if 'T' in keys and paused:
                paused = False
                print('Resume!')
                time.sleep(1)
            elif 'T' in keys and not paused:
                print('Pausing!')
                paused = True
                time.sleep(1)
                print("Total data size: ", get_size('D:/data_m2n'))

            if paused:
                continue

            screen = grab_screen(region=(0, 40, 1600, 920))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = screen[320:, :, :]
            screen = cv2.GaussianBlur(screen, (3, 3), 0)
            screen = cv2.resize(screen, (200, 66), interpolation=cv2.INTER_AREA)

            output = keys_to_output(keys)
            image_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path_name = binary_image_name.format(image_name)

            np.save(image_path_name, screen)
            writer.writerow([image_path_name, *output])
            my_file.flush()
            # break
            # cv2.imshow('window', cv2.resize(screen, (1024, 800)))
            # cv2.imshow('window2', screen)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break


main()
