import argparse
import os.path
import sys

from PyQt5.QtWidgets import QApplication
from platformdirs import user_data_dir, user_config_dir, user_cache_dir
from ultralytics import YOLO

import models.yolo11
import models.yolo11pose
from processing_window import ProcessingWindow
from utils import download_file, get_path_filename, check_files_exist

data_dir = user_data_dir('py-security-metrics', 'xephyrous')
config_dir = user_config_dir('py-security-metrics', 'xephyrous')
cache_dir = user_cache_dir('py-security-metrics', 'xephyrous')

if __name__ == '__main__':
    # CLI Invocation
    args = None
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            prog='csm',
            description='Video security analysis and statistics'
        )

        parser.add_argument('--input', '-i', action='store', dest='input', help='the path of the video to process')
        parser.add_argument('--output', '-o', action='store', dest='output', default='processed.mp4', help='the path to save the processed video')
        parser.add_argument('--stats', '-st', action='store_true', dest='stats', help='outputs the stats of model processing')
        parser.add_argument('--zones', '-z', action='store', dest='zones', help='the zones definition file to use during processing')
        parser.add_argument('--proc-wind', '-pw', action='store_true', dest='proc_wind', help='displays the tile processing debug window during processing')

        model_sizes = parser.add_mutually_exclusive_group()
        model_sizes.add_argument('--nano', '-n', action='store_true', dest='nano', help='specifies the nano model for processing (default)')
        model_sizes.add_argument('--small', '-s', action='store_true', dest='small', help='specifies the small model for processing')
        model_sizes.add_argument('--medium', '-m', action='store_true', dest='medium', help='specifies the medium model for processing')
        model_sizes.add_argument('--large', '-l', action='store_true', dest='large', help='specifies the large model for processing')
        model_sizes.add_argument('--xlarge', '-xl', action='store_true', dest='extra_large', help='specifies the extra large model for processing')

        args = parser.parse_args()

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
                print('Using pretrained-nano model')
            elif args.small:
                check_files_exist('models/pretrained/yolo11s.pt', 'models/pretrained/yolo11s-pose.pt')
                models.yolo11.model = YOLO('models/pretrained/yolo11s.pt')
                print('Using pretrained-small model')
            elif args.medium:
                check_files_exist('models/pretrained/yolo11m.pt', 'models/pretrained/yolo11m-pose.pt')
                models.yolo11.model = YOLO('models/pretrained/yolo11m.pt')
                print('Using pretrained-medium model')
            elif args.large:
                check_files_exist('models/pretrained/yolo11l.pt', 'models/pretrained/yolo11l-pose.pt')
                models.yolo11.model = YOLO('models/pretrained/yolo11l.pt')
                print('Using pretrained-large model')
            elif args.extra_large:
                check_files_exist('models/pretrained/yolo11x.pt', 'models/pretrained/yolo11x-pose.pt')
                models.yolo11.model = YOLO('models/pretrained/yolo11x.pt')
                print('Using pretrained-extra-large model')
            else:
                print('No model type specified!')
                raise FileNotFoundError()
        except FileNotFoundError as e:
            print(f"Error loading specified pretrained model\n{e}Attempting fallback model!")
            try:
                check_files_exist('models/pretrained/yolo11n.pt', 'models/pretrained/yolo11n-pose.pt')
                models.yolo11.model = YOLO('models/pretrained/yolo11n.pt')
                print('Using fallback pretrained-nano model')
            except FileNotFoundError as e:
                print(f"Error loading fallback model: {e}")
                exit(-1)

    # Application entrypoint
    app = QApplication(sys.argv)
    window = ProcessingWindow(args)
    window.setWindowTitle('Py Security Metrics v0.1.a')
    window.setMinimumSize(960, 540)
    window.show()

    app.exec()
