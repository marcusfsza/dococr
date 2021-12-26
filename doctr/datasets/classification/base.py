# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import random
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

from doctr.io.image import tensor_from_pil
from doctr.utils.fonts import get_font

from ..datasets import AbstractDataset


def synthesize_text_img(
    text: str,
    img_size: Optional[Tuple[int, int]] = None,
    font_size: Optional[int] = None,
    font_family: Optional[str] = None,
    background_color: Optional[Tuple[int, int, int]] = None,
    text_color: Optional[Tuple[int, int, int]] = None,
    text_pos: Optional[Tuple[int, int]] = None,
) -> Image:
    """Generate a synthetic character image with black background and white text

    Args:
        text: the character to render as an image
        img_size: the size of the rendered image
        font_size: the size of the font
        font_family: the font family (has to be installed on your system)
        background_color: background color of the final image
        text_color: text color on the final image
        text_pos: offset of the text

    Returns:
        PIL image of the character
    """

    background_color = (0, 0, 0) if background_color is None else background_color
    text_color = (255, 255, 255) if text_color is None else text_color
    default_h = 32
    if font_size is None:
        font_size = int(0.9 * default_h) if img_size is None else int(0.9 * img_size[0])

    font = get_font(font_family, font_size)
    text_size = font.getsize(text)
    if img_size is None:
        img_size = (default_h, text_size[0] if len(text) > 1 else default_h)

    img = Image.new('RGB', img_size[::-1], color=background_color)
    d = ImageDraw.Draw(img)

    # Draw the character
    if text_pos is None:
        text_pos = (0, 0) if text_size[0] >= img_size[1] else (int(round(img_size[0] * 3 / 16)), 0)
    d.text(text_pos, text, font=font, fill=text_color)

    return img


class _CharacterGenerator(AbstractDataset):

    def __init__(
        self,
        vocab: str,
        num_samples: int,
        cache_samples: bool = False,
        font_family: Optional[Union[str, List[str]]] = None,
        img_transforms: Optional[Callable[[Any], Any]] = None,
        sample_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ) -> None:
        self.vocab = vocab
        self._num_samples = num_samples
        self.font_family = font_family if isinstance(font_family, list) else [font_family]  # type: ignore[list-item]
        # Validate fonts
        if isinstance(font_family, list):
            for font in self.font_family:
                try:
                    _ = get_font(font, 10)
                except OSError:
                    raise ValueError(f"unable to locate font: {font}")
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms

        self._data: List[Image.Image] = []
        if cache_samples:
            self._data = [
                (synthesize_text_img(char, font_family=font), idx)
                for idx, char in enumerate(self.vocab) for font in self.font_family
            ]

    def __len__(self) -> int:
        return self._num_samples

    def _read_sample(self, index: int) -> Tuple[Any, int]:
        # Samples are already cached
        if len(self._data) > 0:
            idx = index % len(self._data)
            pil_img, target = self._data[idx]
        else:
            target = index % len(self.vocab)
            pil_img = synthesize_text_img(self.vocab[target], font_family=random.choice(self.font_family))
        img = tensor_from_pil(pil_img)

        return img, target
