import time
import cv2
import tensorflow as tf
import numpy as np
import random

from tensorflow.python.keras.models import load_model
from collections import deque
from statistics import mean

from directkeys import PressKey, ReleaseKey, W, A, S, D
from grabscreen import grab_screen
from getkeys import key_check
from motion import motion_detection


log_len = 25
motion_req = 800
motion_log = deque(maxlen=log_len)


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def count_down():
    for i in list(range(4))[::-1]:
        print("Script start in ", i + 1)
        time.sleep(1)


def img_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img[320:, :, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    img = img / 255
    return img


def main():
    count_down()
    model_path = './models/25_elu_5_dropout-027-0.735988-0.728406.h5'
    print("Loading model", model_path)
    model = load_model(model_path)
    paused = True

    screen = grab_screen(region=(0, 40, 1600, 920))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (200, 66))

    t_minus = prev
    t_now = prev
    t_plus = prev

    while True:
        keys = key_check()
        if 'T' in keys and paused:
            paused = False
            print('Resume!')
            time.sleep(1)
        elif 'T' in keys and not paused:
            print('Pausing!')
            paused = True
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
            time.sleep(1)
        if paused:
            continue

        screen = grab_screen(region=(0, 40, 1600, 920))
        # screen_to_show = screen.copy()
        image_to_predict = screen.copy()
        # screen_to_show = cv2.cvtColor(screen_to_show, cv2.COLOR_BGR2RGB)
        # screen_to_show = cv2.resize(screen_to_show, (1024, 800))

        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.resize(screen, (200, 66))

        # cv2.imshow('window', img_preprocess(image_to_predict))
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        # continue

        last_time = time.time()
        delta_count_last = motion_detection(screen, t_minus, t_now, t_plus)

        t_minus = t_now
        t_now = t_plus
        t_plus = screen
        t_plus = cv2.blur(t_plus, (4, 4))

        prediction = model.predict(img_preprocess(image_to_predict).reshape(1, 66, 200, 3))[0]
        # prediction = np.array(prediction) * np.array(
        #     [2.5, 0.1, 0.1, 0.1, 1.0, 1.0, 0.5, 0.5, 0.1])
        model_choice = np.argmax(prediction)

        if model_choice == 0:
            straight()
            choice_picked = 'straight'

        elif model_choice == 1:
            reverse()
            choice_picked = 'reverse'

        elif model_choice == 2:
            left()
            choice_picked = 'left'
        elif model_choice == 3:
            right()
            choice_picked = 'right'
        elif model_choice == 4:
            forward_left()
            choice_picked = 'forward+left'
        elif model_choice == 5:
            forward_right()
            choice_picked = 'forward+right'
        elif model_choice == 6:
            reverse_left()
            choice_picked = 'reverse+left'
        elif model_choice == 7:
            reverse_right()
            choice_picked = 'reverse+right'
        elif model_choice == 8:
            no_keys()
            choice_picked = 'nokeys'

        motion_log.append(delta_count_last)
        motion_avg = round(mean(motion_log), 3)
        print('loop took {} seconds. Motion: {}. Choice: {}'.format(
            round(time.time() - last_time, 3), motion_avg, choice_picked))

        if motion_avg < motion_req and len(motion_log) >= log_len:
            print('WERE PROBABLY STUCK FFS, initiating some evasive maneuvers.')

            # 0 = reverse straight, turn left out
            # 1 = reverse straight, turn right out
            # 2 = reverse left, turn right out
            # 3 = reverse right, turn left out

            quick_choice = random.randrange(0, 4)

            if quick_choice == 0:
                reverse()
                time.sleep(random.uniform(1, 2))
                forward_left()
                time.sleep(random.uniform(1, 2))

            elif quick_choice == 1:
                reverse()
                time.sleep(random.uniform(1, 2))
                forward_right()
                time.sleep(random.uniform(1, 2))

            elif quick_choice == 2:
                reverse_left()
                time.sleep(random.uniform(1, 2))
                forward_right()
                time.sleep(random.uniform(1, 2))

            elif quick_choice == 3:
                reverse_right()
                time.sleep(random.uniform(1, 2))
                forward_left()
                time.sleep(random.uniform(1, 2))

            for i in range(log_len - 2):
                del motion_log[0]

        # break


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
main()
sess.close()
