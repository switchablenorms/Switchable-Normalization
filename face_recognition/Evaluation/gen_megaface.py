from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import sys
import numpy as np
import argparse
import struct
from PIL import Image
import torchvision.transforms as transforms
import sklearn
from sklearn import preprocessing
sys.path.append("..")
from models.model_builder import ArcFaceWithLoss
from devkit.core import load_state_ckpt


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
trans = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
def read_img(image_path):
  with Image.open(image_path) as img:
    img = img.convert('RGB')
  return trans(img)


def get_feature(imgs, model):
  count = len(imgs)
  imgs_src = []
  imgs_flip = []
  # data = torch.zeros(count * 2, 3, imgs[0].shape[0], imgs[0].shape[1])
  for idx, img in enumerate(imgs):
    for flipid in [0, 1]:
      _img = img
      if flipid == 1:
        _img = torch.flip(_img,[2])
        imgs_flip.append(_img)
        continue
      imgs_src.append(_img)
  imgs_src = torch.stack(imgs_src,0)
  imgs_flip = torch.stack(imgs_flip,0)
  data = torch.cat((imgs_src,imgs_flip),0)
      # data[count * flipid + idx] = _img

  model.eval()
  with torch.no_grad():
    feature = model(data.cuda(), None, extract_mode=True)
  x = feature.data.cpu().numpy()
  embedding = x[0:count, :] + x[count:, :]
  embedding = preprocessing.normalize(embedding)
  return embedding


def write_bin(path, feature):
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature), 1, 4, 5))
    f.write(struct.pack("%df" % len(feature), *feature))


def get_and_write(buffer, model):
  imgs = []
  for k in buffer:
    imgs.append(k[0])
  features = get_feature(imgs, model)
  # print(np.linalg.norm(feature))
  assert features.shape[0] == len(buffer)
  for ik, k in enumerate(buffer):
    out_path = k[1]
    feature = features[ik].flatten()
    write_bin(out_path, feature)


def main(args):
  print(args)

  model = ArcFaceWithLoss(args.backbone, 85742, args.norm_func, args.embedding_size, args.use_se)
  model.cuda()
  load_state_ckpt(args.checkpoint_path, model)

  facescrub_out = os.path.join('./'+args.algo+'_'+args.output, 'facescrub')
  megaface_out = os.path.join('./'+args.algo+'_'+args.output, 'megaface')

  i = 0
  succ = 0
  buffer = []
  for line in open(args.facescrub_lst, 'r'):
    if i % 1000 == 0:
      print("writing fs", i, succ)
    i += 1
    image_path = line.strip()
    _path = image_path.split('/')
    a, b = _path[-2], _path[-1]
    out_dir = os.path.join(facescrub_out, a)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    image_path = os.path.join(args.facescrub_root, image_path)
    img = read_img(image_path)
    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer) == args.batch_size:
      get_and_write(buffer, model)
      buffer = []
    succ += 1
  if len(buffer) > 0:
    get_and_write(buffer, model)
    buffer = []
  print('fs stat', i, succ)

  i = 0
  succ = 0
  buffer = []
  for line in open(args.megaface_lst, 'r'):
    if i % 1000 == 0:
      print("writing mf", i, succ)
    i += 1
    image_path = line.strip()
    _path = image_path.split('/')
    a1, a2, b = _path[-3], _path[-2], _path[-1]
    out_dir = os.path.join(megaface_out, a1, a2)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      # continue
    # print(landmark)
    image_path = os.path.join(args.megaface_root, image_path)
    img = read_img(image_path)
    if img is None:
      print('read error:', image_path)
      continue
    out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer) == args.batch_size:
      get_and_write(buffer, model)
      buffer = []
    succ += 1
  if len(buffer) > 0:
    get_and_write(buffer, model)
    buffer = []
  print('mf stat', i, succ)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', type=int, help='', default=90)
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')
  parser.add_argument('--algo', type=str, help='', default='resnet50sn')
  parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
  parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
  parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
  parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
  parser.add_argument('--output', type=str, help='', default='feature_out')
  parser.add_argument('--backbone', type=str, help='', default='resnet50')
  parser.add_argument('--norm-func', type=str, help='', default='sn')
  parser.add_argument('--embedding-size', type=int, help='', default=512)
  parser.add_argument('--use-se', type=bool, help='', default=False)
  parser.add_argument('--checkpoint-path', type=str, help='', default='')
  return parser.parse_args(argv)


if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
