# -*- coding: utf-8 -*-
"""
Facial Emotion Recognition using ResNet50 with CBAM Attention
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv2D, Layer, Reshape, 
    GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply,
    Activation, BatchNormalization, Add, Concatenate, Lambda
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub

# Download dataset
def download_dataset():
    path = kagglehub.dataset_download("mstjebashazida/affectnet")
    print("Path to dataset files:", path)
    return path

# ------------------ DATASET PREPARATION ------------------

def prepare_datasets(base_path="/kaggle/input/affectnet/archive (3)", batch_size=32, img_size=(224, 224)):
    """
    Prepare training, validation and test datasets with data augmentation
    """
    emotion_classes = ['angry', 'contempt', 'disgust', 'fear',
                       'happy', 'neutral', 'sad', 'surprise']

    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
        validation_split=0.2  # 20% for validation
    )

    # No augmentation for validation/test
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    # Training data (80%)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_path, "Train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=emotion_classes,
        shuffle=True,
        subset='training'
    )

    # Validation data (20%)
    validation_generator = test_datagen.flow_from_directory(
        os.path.join(base_path, "Train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=emotion_classes,
        shuffle=False,
        subset='validation'
    )

    # Test data
    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_path, "Test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=emotion_classes,
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

# ------------------ ATTENTION MODULES ------------------

def channel_attention(input_feature, ratio=8):
    """
    Channel Attention Module focuses on 'what' features to emphasize
    """
    channel = input_feature.shape[-1]

    # Shared MLP
    shared_layer_one = Dense(channel // ratio, activation='relu',
                            kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal',
                            use_bias=True, bias_initializer='zeros')

    # Global Average Pooling
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    # Global Max Pooling
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    # Fusion
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    """
    Spatial Attention Module focuses on 'where' features to emphasize
    """
    kernel_size = 7

    # Average and Max pooling across channel dimension
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)

    # Concatenate both features
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    # Spatial attention with convolution
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size,
                         strides=1, padding='same',
                         activation='sigmoid',
                         kernel_initializer='he_normal',
                         use_bias=False)(concat)

    return Multiply()([input_feature, cbam_feature])

def cbam_block(input_feature, ratio=8):
    """
    CBAM: Convolutional Block Attention Module
    Combines both channel and spatial attention
    """
    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

# ------------------ MODEL ARCHITECTURE ------------------

def build_resattnet_model(num_classes=8, fine_tune_layers=15):
    """
    Build ResNet-based CNN with Attention (ResAttNet)
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,  # Remove top layer to add attention modules
        input_shape=(224, 224, 3)
    )

    # Get output from the last convolutional block
    x = base_model.output

    # Add CBAM attention module after the base ResNet
    x = cbam_block(x)

    # Global pooling and classification layers
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last `fine_tune_layers` layers of the base model
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Make sure all new layers are trainable
    for layer in model.layers:
        if layer not in base_model.layers:
            layer.trainable = True

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ------------------ CALLBACKS ------------------

def get_callbacks():
    """
    Define callbacks for training
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_resattnet_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    return callbacks

# ------------------ TRAINING ------------------

def train_model(model, train_gen, val_gen, callbacks):
    """
    Two-phase training strategy implementation
    """
    # Phase 1: Train only the new layers with frozen base model
    print("Phase 1: Training only the attention and classification layers...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # This is the base_model
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        epochs=5,  # Short initial training
        validation_data=val_gen
    )

    # Phase 2: Fine-tune model with lower learning rate
    print("Phase 2: Fine-tuning the model...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # This is the base_model
            for l in layer.layers[-30:]:  # Unfreeze more layers for fine-tuning
                l.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        epochs=70,  # Set a high epoch limit, early stopping will prevent overfitting
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    return model, history

# ------------------ EVALUATION ------------------

def evaluate_model(model, test_gen):
    """
    Evaluate model performance and visualize results
    """
    print("\nEvaluating on test set:")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}")

    # Predictions
    test_gen.reset()
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return y_true, y_pred_classes, class_names

def plot_training_history(history):
    """
    Plot training history metrics
    """
    # Training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.show()

def calculate_per_class_accuracy(y_true, y_pred_classes, class_names):
    """
    Calculate and display per-class accuracy
    """
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true == i)[0]
        class_accuracy = np.sum(y_pred_classes[class_indices] == i) / len(class_indices)
        per_class_accuracy[class_name] = class_accuracy

    print("\nPer-class Accuracy:")
    for class_name, accuracy in per_class_accuracy.items():
        print(f"{class_name}: {accuracy:.4f}")
    
    return per_class_accuracy

# ------------------ MAIN EXECUTION ------------------

def main():
    # Download dataset
    dataset_path = download_dataset()
    
    # Prepare datasets
    train_gen, val_gen, test_gen = prepare_datasets()
    
    # Build model
    model = build_resattnet_model(num_classes=8, fine_tune_layers=30)
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    model, history = train_model(model, train_gen, val_gen, callbacks)
    
    # Evaluate model
    y_true, y_pred_classes, class_names = evaluate_model(model, test_gen)
    
    # Plot training history
    plot_training_history(history)
    
    # Calculate per-class accuracy
    per_class_accuracy = calculate_per_class_accuracy(y_true, y_pred_classes, class_names)
    
    # Save final model
    model.save("facial_expression_resattnet.h5")
    
    print("Model saved successfully!")
    
    return model

# Run everything
if __name__ == "__main__":
    # Install required packages
    try:
        import tensorflow
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "tensorflow"])
        
    try:
        import cv2
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "opencv-python"])
    
    model = main()
