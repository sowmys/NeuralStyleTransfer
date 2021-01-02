"""
Calculates loss for the content image and style images
"""

import tensorflow as tf


class LossCalculator:
    content_weight: float
    style_weight: float
    total_variation_weight: float
    target_image_rows: int
    target_image_cols: int

    def __init__(self,
                 target_image_rows: int,
                 target_image_cols: int,
                 content_weight: float = 2.5e-8,
                 style_weight: float = 1e-6,
                 total_variation_weight: float = 1e-6):
        self.target_image_rows = target_image_rows
        self.target_image_cols = target_image_cols
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight

    def style_loss(self, style: tf.Tensor, combination: tf.Tensor) -> tf.Tensor:
        style_gram_matrix: tf.Tensor = self.gram_matrix(style)
        content_gram_matrix: tf.Tensor = self.gram_matrix(combination)
        channels: int = 3
        size: int = self.target_image_rows * self.target_image_cols
        style_loss_square_sum = tf.reduce_sum(tf.square(style_gram_matrix - content_gram_matrix))
        return self.style_weight * style_loss_square_sum / (4.0 * (channels ** 2) * (size ** 2))

    def content_loss(self, base: tf.Tensor, combination: tf.Tensor) -> tf.Tensor:
        return self.content_weight * tf.reduce_sum(tf.square(combination - base))

    def total_variation_loss(self, image: tf.Tensor) -> tf.Tensor:
        # calculate high pass
        x_variation = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_variation = image[:, 1:, :, :] - image[:, :-1, :, :]
        total_variation_sum = tf.reduce_sum(tf.abs(x_variation)) + tf.reduce_sum(tf.abs(y_variation))
        return self.total_variation_weight * total_variation_sum

    @staticmethod
    def gram_matrix(x: tf.Tensor) -> tf.Tensor:
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram
