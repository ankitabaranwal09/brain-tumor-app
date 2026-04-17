import tensorflow as tf
from tensorflow.keras.layers import InputLayer, BatchNormalization
from tensorflow.keras.mixed_precision import Policy

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)
        super().__init__(*args, **kwargs)

class PatchedBatchNormalization(BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop("synchronized", None)
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)

custom_objects = {
    "InputLayer": PatchedInputLayer,
    "BatchNormalization": PatchedBatchNormalization,
    "DTypePolicy": Policy,
}

model = tf.keras.models.load_model(
    "model/brain_tumor_model.h5",
    custom_objects=custom_objects,
    compile=False
)

model.save("model/brain_tumor_model_portable.keras")

print("Portable model saved successfully!")