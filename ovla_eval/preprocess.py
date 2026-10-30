"""Tiền xử lý ảnh, sao chép đúng pipeline của OpenVLA (experiments/robot/{libero_utils,openvla_utils}.py).

Model được train trên ảnh RLDS (JPEG 256x256 -> resize 224 bằng lanczos3) với augmentation random-crop
90% diện tích, nên lúc infer phải resize giống hệt và center-crop 90%. Lệch bước này làm tụt success rate rõ rệt.
"""

import numpy as np
import tensorflow as tf

try:
    # TF chỉ dùng để xử lý ảnh / đọc RLDS, nhường toàn bộ GPU cho torch
    tf.config.set_visible_devices([], "GPU")
except (RuntimeError, ValueError):
    pass

IMAGE_SIZE = 224
CENTER_CROP_SCALE = 0.9


def resize_image(img: np.ndarray, size: int = IMAGE_SIZE, jpeg_roundtrip: bool = True) -> np.ndarray:
    """Resize về size x size như lúc train.

    jpeg_roundtrip=True cho ảnh render từ simulator (giả lập việc RLDS lưu ảnh dạng JPEG);
    False cho ảnh đọc từ RLDS vì tfds đã decode từ JPEG rồi.
    """
    x = tf.convert_to_tensor(img, dtype=tf.uint8)
    if jpeg_roundtrip:
        x = tf.io.decode_image(tf.image.encode_jpeg(x), expand_animations=False, dtype=tf.uint8)
    x = tf.image.resize(x, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy()


def center_crop(img: np.ndarray, crop_scale: float = CENTER_CROP_SCALE) -> np.ndarray:
    """Crop giữa ảnh với diện tích = crop_scale rồi resize lại kích thước cũ (tf.image.crop_and_resize như OpenVLA)."""
    h, w = img.shape[:2]
    side = float(np.clip(np.sqrt(crop_scale), 0.0, 1.0))
    off = (1.0 - side) / 2.0
    x = tf.image.convert_image_dtype(tf.convert_to_tensor(img, dtype=tf.uint8), tf.float32)[None]
    boxes = tf.constant([[off, off, off + side, off + side]], dtype=tf.float32)
    x = tf.image.crop_and_resize(x, boxes, tf.constant([0], dtype=tf.int32), (h, w))
    x = tf.clip_by_value(x[0], 0.0, 1.0)
    return tf.image.convert_image_dtype(x, tf.uint8, saturate=True).numpy()
