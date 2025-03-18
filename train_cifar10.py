import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

def load_and_preprocess_data():
    """Load CIFAR-10 data, split into train/validation/test, and preprocess."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Let's reserve 5,000 images from x_train for validation
    x_val = x_train[-5000:]
    y_val = y_train[-5000:]
    x_train = x_train[:-5000]
    y_train = y_train[:-5000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def resize_and_preprocess_image(image, label):
    """Resize image to (224, 224) and preprocess for MobileNetV2."""
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)  # scales pixels between -1 and 1
    return image, label

def create_tf_dataset(x, y, batch_size=32, shuffle=False):
    """Create a tf.data.Dataset and apply resizing/preprocessing."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.map(
        lambda img, lbl: resize_and_preprocess_image(img, lbl),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)

def build_model(num_classes=10):
    """Build a transfer learning model using MobileNetV2 as the base."""
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # Environment variables for hyperparameters
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    epochs = int(os.environ.get("EPOCHS", 5))

    # 1. Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    print("Data shapes:")
    print(f"  Train set: {x_train.shape}, {y_train.shape}")
    print(f"  Val set:   {x_val.shape}, {y_val.shape}")
    print(f"  Test set:  {x_test.shape}, {y_test.shape}")

    # 2. Create tf.data.Dataset objects
    train_ds = create_tf_dataset(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds   = create_tf_dataset(x_val,   y_val,   batch_size=batch_size, shuffle=False)
    test_ds  = create_tf_dataset(x_test,  y_test,  batch_size=batch_size, shuffle=False)

    # 3. Build model
    model = build_model(num_classes=10)

    # 4. Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 5. Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # 6. Save model
    model.save("/app/saved_model")
    print("Model saved to /app/saved_model")

if __name__ == "__main__":
    main()
