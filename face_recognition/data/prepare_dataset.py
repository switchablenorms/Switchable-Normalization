import argparse
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import cv2
from pathlib import Path

def load_mx_rec(rec_path):
  save_path = rec_path / 'imgs'
  if not save_path.exists():
    save_path.mkdir()
  list_writer = open(str(rec_path / 'train.txt'),'w+')
  imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'), str(rec_path / 'train.rec'), 'r')
  img_info = imgrec.read_idx(0)
  header, _ = mx.recordio.unpack(img_info)
  max_idx = int(header.label[0])
  for idx in tqdm(range(1, max_idx)):
    img_info = imgrec.read_idx(idx)
    header, img = mx.recordio.unpack_img(img_info)
    label = int(header.label)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    label_path = save_path / str(label)
    if not label_path.exists():
      label_path.mkdir()
    img.save(label_path / '{}.jpg'.format(idx), quality=95)
    list_writer.writelines(str(label) + '/{}.jpg'.format(idx)+' '+str(label))
  list_writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='unpack mxnet record')
  parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='faces_emore', type=str)
  args = parser.parse_args()
  rec_path = Path(args.rec_path)
  load_mx_rec(rec_path)

