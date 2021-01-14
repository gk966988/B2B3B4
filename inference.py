import torch
from config import cfg
from models.net import choose_net
import torchvision.transforms as T
from dataset import MyData
from torch.utils.data import DataLoader
from utils.utils import get_transform
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
from PIL import Image

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
config_files = ['../input/b2b3b4-1/configs/efficientnetb2.yaml', '../input/b2b3b4-1/configs/efficientnetb3.yaml', '../input/b2b3b4-1/configs/efficientnetb4.yaml']



def load_model(config_file, imgs_path):
    outputs = None
    # for config_file in config_files:
    cfg.merge_from_file(config_file)
    model = choose_net(name=cfg.MODEL.NAME, num_classes=cfg.MODEL.CLASSES, weight_path=cfg.MODEL.WEIGHT_FROM)
    
    # weight_path = cfg.MODEL.MODEL_PATH + cfg.MODEL.NAME + '.pth'
    weight_path = '../input/b2b3b4-1/weights/'+ cfg.MODEL.NAME + '.pth'
    checkpoint = torch.load(weight_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    for img_path in imgs_path:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output, _ = model(img)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=0)
    return outputs, imgs_name


if __name__=='__main__':

    import os
    from os.path import join
    from tqdm import tqdm
    import pandas as pd

    root = '/kaggle/input/cassava-leaf-disease-classification'
    path_dir = join(root, 'test_images')
    imgs_name = os.listdir(path_dir)
    imgs_path = [os.path.join(path_dir, e) for e in imgs_name]
    pred = torch.zeros(size=[len(imgs_name), 5])
    for config_file in config_files:
        output, imgs_name = load_model(config_file, imgs_path)
        pred += output.cpu()/3
    pred = F.softmax(pred, dim=1).cpu().numpy()
    pred = pred.argmax(1)
    sub = pd.DataFrame({'image_id': imgs_name, 'label': pred})
    # print(sub)
    sub.to_csv("submission.csv", index=False)









