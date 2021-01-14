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


config_files = ['./configs/efficientnetb2.yaml', './configs/efficientnetb3.yaml', './configs/efficientnetb4.yaml']



def load_model(config_file, imgs_path):
    outputs = None
    # for config_file in config_files:
    cfg.merge_from_file(config_file)
    model = choose_net(name=cfg.MODEL.NAME, num_classes=cfg.MODEL.CLASSES, weight_path=cfg.MODEL.WEIGHT_FROM)
    weight_path = cfg.MODEL.MODEL_PATH + cfg.MODEL.NAME + '.pth'
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model.cuda()
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
        pred += output/3
    pred = F.softmax(pred, dim=1).cpu().numpy()
    pred = pred.argmax(1)
    sub = pd.DataFrame({'image_id': imgs_name, 'label': pred})
    # print(sub)
    sub.to_csv("submission.csv", index=False)
















# test_dataset = MyData(root=cfg.DATASETS.ROOT_TEST, phase='test',
#                              transform=get_transform(cfg.INPUT.SIZE_TRAIN, 'test'))
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
#                              num_workers=2, pin_memory=True)




# image_name = []
# outputs = []
#
# num = 0
# for config_file in config_files:
#     num += 1
#     preds = []
#     cfg.merge_from_file(config_file)
#     model = choose_net(name=cfg.MODEL.NAME, num_classes=cfg.MODEL.CLASSES, weight_path=cfg.MODEL.WEIGHT_FROM)
#     weight_path = cfg.MODEL.MODEL_PATH + cfg.MODEL.NAME + '.pth'
#     state_dict = torch.load(weight_path)
#     model.load_state_dict(state_dict)
#     print('Network loaded from {}'.format(weight_path))
#     model.cuda()
#     model.eval()
#     with torch.no_grad():
#         pbar = tqdm(enumerate(test_loader), total=int(len(test_loader.dataset)))
#         pbar.set_description('Validation')
#         for batch_idx, (img, _, img_name) in pbar:
#             img = img.cuda()
#             y_pred, y_metric  = model(img)
#             # pred = F.softmax(y_pred, dim=1).cpu().numpy()
#             pred = y_pred.cpu().numpy()
#             preds.append(pred)
#             output = np.concatenate(preds, 0)
#             if num == len(config_files):
#                 image_name.append(img_name[0])
#     outputs.append(output)
#
# outputs = np.concatenate(outputs, 1)


















