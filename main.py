import os.path

from PyQt5.QtWidgets import QApplication
from ultralytics import YOLO

import models.yolo11
import models.yolo11pose
from utils import download_file, get_path_filename, check_files_exist
from window import Window
import sys, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Py Security Metrics v0.2.a',
        description=''
    )

    parser.add_argument('--input', '-i', action='store', dest='input', help='The path of the video to process')
    parser.add_argument('--output', '-o', action='store', dest='output', default='processed.mp4', help='The path to save the processed video')
    parser.add_argument('--stats', '-st', action='store_true', dest='stats', help='Outputs the stats of model processing')

    model_sizes = parser.add_mutually_exclusive_group()
    model_sizes.add_argument('--nano', '-n', action='store_true', dest='nano', help='')
    model_sizes.add_argument('--small', '-s', action='store_true', dest='small')
    model_sizes.add_argument('--medium', '-m', action='store_true', dest='medium')
    model_sizes.add_argument('--large', '-l', action='store_true', dest='large')
    model_sizes.add_argument('--xlarge', '-xl', action='store_true', dest='extra_large')

    args = parser.parse_args()
    app = QApplication(sys.argv)

    # Download missing pretrained models
    with open('models/pretrained/download_list.txt') as file:
        for url in file:
            url = url.strip().replace('\n', '')
            file_path = 'models/pretrained/' + get_path_filename(url)

            if not os.path.exists(file_path):
                download_file(url, 'models/pretrained/' + get_path_filename(url))

    # Load selected models
    try:
        if args.nano:
            check_files_exist('models/pretrained/yolo11n.pt', 'models/pretrained/yolo11n-pose.pt')
            models.yolo11.model = YOLO('models/pretrained/yolo11n.pt')
            models.yolo11pose.model = YOLO('models/pretrained/yolo11n-pose.pt')
            print('Using pretrained-nano model')
        elif args.small:
            check_files_exist('models/pretrained/yolo11s.pt', 'models/pretrained/yolo11s-pose.pt')
            models.yolo11.model = YOLO('models/pretrained/yolo11s.pt')
            models.yolo11pose.model = YOLO('models/pretrained/yolo11s-pose.pt')
            print('Using pretrained-small model')
        elif args.medium:
            check_files_exist('models/pretrained/yolo11m.pt', 'models/pretrained/yolo11m-pose.pt')
            models.yolo11.model = YOLO('models/pretrained/yolo11m.pt')
            models.yolo11pose.model = YOLO('models/pretrained/yolo11m-pose.pt')
            print('Using pretrained-medium model')
        elif args.large:
            check_files_exist('models/pretrained/yolo11l.pt', 'models/pretrained/yolo11l-pose.pt')
            models.yolo11.model = YOLO('models/pretrained/yolo11l.pt')
            models.yolo11pose.model = YOLO('models/pretrained/yolo11l-pose.pt')
            print('Using pretrained-large model')
        elif args.extra_large:
            check_files_exist('models/pretrained/yolo11x.pt', 'models/pretrained/yolo11x-pose.pt')
            models.yolo11.model = YOLO('models/pretrained/yolo11x.pt')
            models.yolo11pose.model = YOLO('models/pretrained/yolo11x-pose.pt')
            print('Using pretrained-extra-large model')
        else:
            print('No model type specified!')
            raise FileNotFoundError()
    except FileNotFoundError as e:
        print(f"Error loading specified pretrained model\n{e}Attempting fallback model!")
        try:
            check_files_exist('models/pretrained/yolo11n.pt', 'models/pretrained/yolo11n-pose.pt')
            models.yolo11.model = YOLO('models/pretrained/yolo11n.pt')
            models.yolo11pose.model = YOLO('models/pretrained/yolo11n-pose.pt')
            print('Using fallback pretrained-nano model')
        except FileNotFoundError as e:
            print(f"Error loading fallback model: {e}")
            exit(-1)

    # Application entrypoint
    window = Window(args)
    window.setWindowTitle('Py Security Metrics v0.1.a')
    window.setFixedSize(960, 540)
    window.show()

    app.exec()
