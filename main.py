from tensorflow import keras

from image_processor import ImageProcessor
from loss_calculator import LossCalculator
from neural_style_transfer_model import NeuralStyleTransferModel


base_image_path: str = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_reference_image_path: str = keras.utils.get_file("starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg")
result_prefix: str = "paris_generated"

# Dimensions of the generated picture.
target_image_rows: int = 400
target_image_cols: int = ImageProcessor.calculate_target_cols(base_image_path, target_image_rows)

optimizer: keras.optimizers.Optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0,
        decay_steps=100,
        decay_rate=0.96
    )
)
loss_calculator = LossCalculator(target_image_rows, target_image_cols)
image_processor = ImageProcessor(target_image_rows, target_image_cols)

nst_model = NeuralStyleTransferModel(target_image_rows, target_image_cols, loss_calculator, optimizer, image_processor)

nst_model.generate(base_image_path, style_reference_image_path, result_prefix, iterations=500)
