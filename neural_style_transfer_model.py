"""
A keras model that implements the neural style transfer
"""
from typing import List, ClassVar, Dict, Any, Tuple
from image_processor import ImageProcessor
import tensorflow as tf

from loss_calculator import LossCalculator


class NeuralStyleTransferModel:
    style_layer_names: ClassVar[List[str]] = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    content_layer_name: ClassVar[str] = "block5_conv2"

    target_image_rows: int
    target_image_cols: int
    image_processor: ImageProcessor
    optimizer: tf.keras.optimizers.Optimizer
    loss_calculator: LossCalculator
    feature_extractor: tf.keras.Model

    def __init__(self,
                 target_image_rows: int,
                 target_image_cols: int,
                 loss_calculator: LossCalculator,
                 optimizer: tf.keras.optimizers.Optimizer,
                 image_processor: ImageProcessor) -> None:
        self.target_image_rows = target_image_rows
        self.target_image_cols = target_image_cols
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.image_processor = image_processor
        # Build a VGG19 model loaded with pre-trained ImageNet weights
        vgg_model: tf.keras.Model = tf.keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)

        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict: Dict[str, Any] = dict([(layer.name, layer.output) for layer in vgg_model.layers])

        # Set up a model that returns the activation values for every layer in
        # VGG19 (as a dict).
        self.feature_extractor = tf.keras.Model(inputs=vgg_model.inputs, outputs=outputs_dict)

    def generate(self, base_image_path: str, style_image_path: str, result_file_prefix: str,
                 iterations: int = 4000, generated_image_count: int = 10):

        base_image: tf.Tensor = self.image_processor.load_image(base_image_path)
        style_reference_image: tf.Tensor = self.image_processor.load_image(style_image_path)
        combination_image_var: tf.Variable = tf.Variable(self.image_processor.load_image(base_image_path))
        generation_interval: int = int(iterations / generated_image_count)

        for i in range(1, iterations + 1):
            loss: float
            grads: tf.Tensor
            loss, grads = self.compute_loss_and_grads(combination_image_var, base_image, style_reference_image)
            self.optimizer.apply_gradients([(grads, combination_image_var)])
            if i % generation_interval == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))
                self.image_processor.save_image(combination_image_var.numpy(),
                                                file_name=result_file_prefix + "_at_iteration_%d.png" % i)

    # @tf.function
    def compute_loss_and_grads(
            self,
            combined_image: tf.Variable,
            content_image: tf.Tensor,
            style_image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            input_tensor: tf.Tensor = tf.concat([content_image, style_image, combined_image], axis=0)
            features: tf.Tensor = self.feature_extractor(input_tensor)
            # Add content loss
            layer_features: tf.Tensor = features[self.content_layer_name]
            base_image_features: tf.Tensor = layer_features[0, :, :, :]
            combination_features: tf.Tensor = layer_features[2, :, :, :]
            total_loss: tf.Tensor = tf.zeros(())
            total_loss = total_loss + self.loss_calculator.content_loss(base_image_features, combination_features)
            # Add style loss
            for layer_name in self.style_layer_names:
                layer_features = features[layer_name]
                style_reference_features: tf.Tensor = layer_features[1, :, :, :]
                combination_features = layer_features[2, :, :, :]
                style_loss: tf.Tensor = self.loss_calculator.style_loss(style_reference_features, combination_features)
                total_loss += (style_loss / len(self.style_layer_names))
            # Add total variation loss
            total_loss += self.loss_calculator.total_variation_loss(tf.convert_to_tensor(combined_image))
        computed_gradient: tf.Tensor = tape.gradient(total_loss, combined_image)
        return total_loss, computed_gradient
