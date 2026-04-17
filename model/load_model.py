import tensorflow as tf

base_model = tf.keras.applications.EfficientNetB0(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.load_weights("model/brain_tumor_weights.h5")

print("✅ Model Loaded Successfully")