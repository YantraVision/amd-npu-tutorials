#!/usr/bin/env python

from typing import Tuple

import argparse
import onnxruntime
import os
import sys
import time
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--onnx_model", default="model.onnx", help="Input onnx model")
parser.add_argument(
    "--data_dir",
    default="/workspace/dataset/imagenet",
    help="Directory of dataset")
parser.add_argument(
    "--batch_size", default=1, type=int, help="Evaluation batch size")
parser.add_argument(
    "--ipu",
    action="store_true",
    help="Use IPU for inference.",
)
parser.add_argument(
    "--provider_config",
    type=str,
    default="vaip_config.json",
    help="Path of the config file for seting provider_options.",
)
args = parser.parse_args()

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def accuracy(output: torch.Tensor,
             target: torch.Tensor,
             topk: Tuple[int] = (1,)) -> Tuple[float]:
  """Computes the accuracy over the k top predictions for the specified values of k.
  Args:
    output: Prediction of the model.
    target: Ground truth labels.
    topk: Topk accuracy to compute.
  Returns:
    Accuracy results according to 'topk'.
  """

  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def prepare_data_loader(data_dir: str,
                        batch_size: int = 100,
                        workers: int = 8) -> torch.utils.data.DataLoader:
  """Returns a validation data loader of ImageNet by given `data_dir`.
  Args:
    data_dir: Directory where images stores. There must be a subdirectory named
      'validation' that stores the validation set of ImageNet.
    batch_size: Batch size of data loader.
    workers: How many subprocesses to use for data loading.
  Returns:
    An object of torch.utils.data.DataLoader.
  """

  valdir = os.path.join(data_dir, 'validation')

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  val_dataset = datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ]))

  return torch.utils.data.DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=workers,
      pin_memory=True)
install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
def val_imagenet():
  """Validate ONNX model on ImageNet dataset."""
  print(f'Current onnx model: {args.onnx_model}')
  if args.ipu:
    providers = ["VitisAIExecutionProvider"]
    xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
    provider_options = [{
         'enable_cache_file_io_in_mem': 0,
         'target': 'X1',
         'xlnx_enable_py3_round': 0,
         'xclbin': xclbin_file,
     }]
 
  else:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    provider_options = None
  ort_session = onnxruntime.InferenceSession(
      args.onnx_model, providers=providers, provider_options=provider_options)

  val_loader = prepare_data_loader(args.data_dir, args.batch_size)

  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')

  start_time = time.time()
  val_loader = tqdm(val_loader, file=sys.stdout)
  with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
      images = torch.permute(images, (0, 2, 3, 1))
      inputs, targets = images.numpy(), targets
      ort_inputs = {ort_session.get_inputs()[0].name: inputs}

      outputs = ort_session.run(None, ort_inputs)
      outputs = torch.from_numpy(outputs[0])

      acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
      top1.update(acc1, images.size(0))
      top5.update(acc5, images.size(0))

    current_time = time.time()
    print('Test Top1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'.format(
        float(top1.avg), float(top5.avg), (current_time - start_time)))

  return top1.avg, top5.avg

if __name__ == '__main__':
  val_imagenet()
