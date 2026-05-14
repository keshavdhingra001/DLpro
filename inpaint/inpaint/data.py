import math
import os

import numpy as np
import PIL.Image as Image
import torch
import torchvision as tv
import jsonlines
import time

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class _Flipper:
    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, x):
        if np.random.binomial(1, self._p):
            x = (x[0].transpose(Image.FLIP_LEFT_RIGHT), x[1][::-1].copy())
        return x


class _Jitter:
    def __init__(self, *args, **kwargs):
        self._jitter = tv.transforms.ColorJitter(*args, **kwargs)

    def __call__(self, x):
        return self._jitter(x[0]), x[1]


def _final_transform(x):
    assert x[0].size[0] == x[0].size[1]
    assert not math.modf(math.log2(x[0].size[0]))[0]
    return np.array(x[0], dtype=np.float32).transpose((2, 0, 1)) / 255.0, x[1]


def _generate_dummy_mask(shape):
    return np.ones(shape, dtype=np.float32)


def _generate_random_mask(shape):
    return (np.random.rand(*shape) > 0.2).astype('float32')


class _MaskGenerator:
    def __init__(self, masks_dir, n_images_by_file=10000,
                 min_stroke_width=5, max_stroke_width=15,
                 fat_stroke_width=40, fat_stroke_prob=0.2):
        self.strokes = []
        for name in os.listdir(masks_dir):
            reader = jsonlines.open(os.path.join(masks_dir, name))
            i = 0
            for item in reader:
                self.strokes.extend(item['drawing'])
                i += 1
                if i > n_images_by_file:
                    break
        self.min_stroke_width = min_stroke_width
        self.max_stroke_width = max_stroke_width
        self.fat_stroke_width = fat_stroke_width
        self.fat_stroke_prob = fat_stroke_prob
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        else:
            self.rng.seed(int(time.time() * 100) % (2 ** 32 - 1))

        target_width, target_height = shape
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')

        def central_crop(image, new_width, new_height):
            image = image.copy()
            width, height = image.size
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            image = image.crop((left, top, right, bottom))
            return image

        if self.rng.choice([True, False], p=[self.fat_stroke_prob, 1 - self.fat_stroke_prob]):
            stroke_index = self.rng.randint(0, len(self.strokes))
            stroke_width = self.fat_stroke_width
            stroke = self.strokes[stroke_index][:int(len(self.strokes[stroke_index]) * 1)]
            ax.plot(*stroke, color='black', lw=stroke_width)
        else:
            n_strokes = int(self.rng.normal(5, 1))
            for _ in range(n_strokes):
                stroke_index = self.rng.randint(0, len(self.strokes))
                stroke_width = self.rng.randint(self.min_stroke_width, self.max_stroke_width)
                stroke = self.strokes[stroke_index][:int(len(self.strokes[stroke_index]) * 1)]
                ax.plot(*stroke, color='black', lw=stroke_width)

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        image = Image.fromarray(image)
        image = image.resize((int(1.25 * target_width), int(1.25 * target_height)))
        image = central_crop(image, target_width, target_height)
        image = image.convert('1')
        return np.array(image).astype('float32')


class _CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, masks_dir, transform=_final_transform, val=False,
                 min_stroke_width=5, max_stroke_width=15,
                 fat_stroke_width=40, fat_stroke_prob=0.2):
        self._input = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
        self._transform = transform
        if masks_dir is not None:
            self._mask_generator = _MaskGenerator(
                masks_dir,
                min_stroke_width=min_stroke_width, max_stroke_width=max_stroke_width,
                fat_stroke_width=fat_stroke_width, fat_stroke_prob=fat_stroke_prob
            )
        else:
            self._mask_generator = _generate_random_mask
        self.val = val

    def __len__(self):
        return len(self._input)

    def __getitem__(self, index):
        img = Image.open(self._input[index])
        if self.val:
            mask = self._mask_generator(img.size[::-1], seed=index)
        else:
            mask = self._mask_generator(img.size[::-1])
        return self._transform((img, mask))


def _make_default_transform():
    return tv.transforms.Compose([
        _Flipper(),
        _Jitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        _final_transform
    ])


def make_dataloader(input_dir, masks_dir, transform='default', val=False,
                    min_stroke_width=5, max_stroke_width=15,
                    fat_stroke_width=40, fat_stroke_prob=0.2,
                    *args, **kwargs):
    """Make data loader for training."""
    if transform == 'default':
        transform = _make_default_transform()
    dataset = _CelebaDataset(
        input_dir, masks_dir, transform, val=val,
        min_stroke_width=min_stroke_width, max_stroke_width=max_stroke_width,
        fat_stroke_width=fat_stroke_width, fat_stroke_prob=fat_stroke_prob
    )
    return torch.utils.data.DataLoader(dataset, *args, **kwargs)
