import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
scale=1.1

transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)

for i in range(1,98):
    image=np.array(Image.open('data/vinfast/vinfast ({}).jpg'.format(i)).convert('RGB'))
    image_transformed=transforms(image=image,bboxes=[])
    img=image_transformed['image'].numpy().transpose(1,2,0)
    plt.imsave('data/data augmentation/vinfast_transform_{}.jpg'.format(i),img)
