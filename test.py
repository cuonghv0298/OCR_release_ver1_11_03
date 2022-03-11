import argparse 
import matplotlib.pyplot as plt
from PIL import Image
from traitlets.traitlets import default
import yaml
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def main(config_path, img_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    detector = Predictor(config)
    img = Image.open(img_path)
    s = detector.predict(img)
    print(s)
    return s

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='inference.yaml', help = 'path to config file')
    parser.add_argument('--img_path', type=str, default='sample/test84.png', help = 'path to img')
    args = parser.parse_args()

    main(args.config, args.img_path)