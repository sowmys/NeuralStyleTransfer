"""
Preprocesses images, converts image to Tensor or Numpy Array and back to Image
"""
from typing import Optional, Any, Tuple
import numpy as np
import tensorflow as tf


class ImageProcessor:
    target_image_rows: int
    target_image_cols: int

    def __init__(self, target_image_rows: int, target_image_cols: int):
        self.target_image_rows = target_image_rows
        self.target_image_cols = target_image_cols

    def load_image(self, file_name: str) -> tf.Tensor:
        """
        + Loads an image
        + Convert the image to numpy array
        + Insert a dimension to hold the index that that point to ContentImage(0), StyleImage(1) or CombinedImage(2)
        + preprocesses as follow:
            + The images are converted from RGB to BGR,
            + Each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
        + convert the numpy array to Tensor
        :return:
            Tensor representation of the preprocessed image whose path is provided
        """
        # Util function to open, resize and format pictures into appropriate tensors
        target_size: Tuple[int, int] = (self.target_image_rows, self.target_image_cols)
        img: Optional[Any] = tf.keras.preprocessing.image.load_img(file_name, target_size=target_size)
        nd_array: np.ndarray = tf.keras.preprocessing.image.img_to_array(img)
        expanded_nd_array: np.ndarray = np.expand_dims(nd_array, axis=0)
        preprocessed_nd_array = tf.keras.applications.vgg19.preprocess_input(expanded_nd_array)
        preprocessed_tensor: tf.Tensor = tf.convert_to_tensor(preprocessed_nd_array)
        return preprocessed_tensor

    def save_image(self, image_to_save: np.ndarray, file_name: str) -> None:
        """
        Saves the image after undoing the work done by load_image:
            + Removes the inserted dimension (by reshaping to row, col, 3)
            + Centers by mean (TBD: Need more info on the math of adding constants for each channel)
            + Reverses the BGR to RGB
            + clip RGB value to 0 to 255
        """
        reshaped_image: np.ndarray = image_to_save.reshape((self.target_size[0], self.target_size[1], 3))
        # Remove zero-center by mean pixel
        reshaped_image[:, :, 0] += 103.939
        reshaped_image[:, :, 1] += 116.779
        reshaped_image[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        rgb_image: np.ndarray = reshaped_image[:, :, ::-1]
        clipped_image: np.ndarray = np.clip(rgb_image, 0, 255).astype("uint8")
        tf.keras.preprocessing.image.save_img(file_name, clipped_image)

    @staticmethod
    def calculate_target_cols(base_image_path: str, target_image_rows: int) -> int:
        source_image_width: int
        source_image_height: int
        source_image_width, source_image_height = tf.keras.preprocessing.image.load_img(base_image_path).size
        target_image_cols: int = int(source_image_width * target_image_rows / source_image_height)
        return target_image_cols
