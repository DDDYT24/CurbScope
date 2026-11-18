"""Three text recognisers behind one interface: crop (RGB array) -> text.

- Tesseract (classic LSTM engine), page segmentation mode 7: one line per crop.
- EasyOCR's recogniser (CRNN), given the whole crop as one box.
- TrOCR base-printed (transformer encoder-decoder), pinned to one model commit.
"""

import cv2
import numpy as np
from PIL import Image


def tesseract(config: str):
    import pytesseract

    def read(crop_rgb: np.ndarray) -> str:
        return pytesseract.image_to_string(Image.fromarray(crop_rgb), config=config).strip()

    return read


def easyocr_recognizer(reader):
    def read(crop_rgb: np.ndarray) -> str:
        grey = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        return " ".join(reader.recognize(grey, detail=0)).strip()

    return read


def trocr(model_name: str, revision: str):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(model_name, revision=revision)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, revision=revision).eval()

    def read(crop_rgb: np.ndarray) -> str:
        pixels = processor(images=Image.fromarray(crop_rgb), return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(pixels, max_new_tokens=32)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return read
