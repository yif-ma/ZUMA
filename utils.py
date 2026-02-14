import torchvision.transforms as transforms
from AnomalyCLIP_lib.transform import image_transform
from AnomalyCLIP_lib.constants import *
import math

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = []
    t = image_transform(args.image_size,is_train=False,mean=OPENAI_DATASET_MEAN,std=OPENAI_DATASET_STD)
    t.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size),interpolation=transforms.InterpolationMode.BICUBIC,max_size=None,antialias=None)
    t.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    preprocess.append(t)
    t = image_transform(int(math.ceil(args.image_size * CMPL_SCALE)),is_train=False,mean=OPENAI_DATASET_MEAN,std=OPENAI_DATASET_STD)
    t.transforms[0] = transforms.Resize(size=(int(math.ceil(args.image_size * CMPL_SCALE)), int(math.ceil(args.image_size * CMPL_SCALE))),interpolation=transforms.InterpolationMode.BICUBIC,max_size=None,antialias=None)
    t.transforms[1] = transforms.CenterCrop(size=(int(math.ceil(args.image_size * CMPL_SCALE)), int(math.ceil(args.image_size * CMPL_SCALE))))
    preprocess.append(t)
    target_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.CenterCrop(args.image_size), transforms.ToTensor()])
    target_transform_pc = transforms.Compose([transforms.Resize((args.point_size, args.point_size)), transforms.CenterCrop(args.point_size), transforms.ToTensor()])

    return preprocess, target_transform, target_transform_pc