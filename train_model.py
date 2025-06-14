'''import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from datetime import datetime
import pickle

class GestureModelTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.history = None
        self.model_path = 'models/gesture_model.h5'
        
    def load_data(self):
        """Load all collected data"""
        features_files = glob.glob('data/raw/features_*.npy')
        labels_files = glob.glob('data/raw/labels_*.npy')
        
        if not features_files:
            raise ValueError("No data files found! Please collect data first.")
        
        all_features = []
        all_labels = []
        
        for feat_file, label_file in zip(sorted(features_files), sorted(labels_files)):
            features = np.load(feat_file)
            labels = np.load(label_file)
            all_features.append(features)
            all_labels.extend(labels)
        
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        print(f"Loaded {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        print(f"Unique gestures: {np.unique(y)}")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2):
        """Preprocess and split data"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=42, stratify=y_encoded
        )
        
        # Convert to categorical for neural network
        num_classes = len(np.unique(y_encoded))
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)
        
        return X_train, X_test, y_train_cat, y_test_cat, y_train, y_test
    
    def build_advanced_model(self, input_shape, num_classes):
        """Build advanced neural network with attention mechanism"""
        inputs = keras.Input(shape=(input_shape,))
        
        # Feature extraction layers
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(256, activation='tanh')(x)
        attention = layers.Dense(256, activation='softmax')(attention)
        x = layers.Multiply()([x, attention])
        
        # Classification layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer with learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), 
                    keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, epochs=100, batch_size=32):
        """Train the model"""
        # Load and preprocess data
        X, y = self.load_data()
        X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = self.preprocess_data(X, y)
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_advanced_model(X_train.shape[1], num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        print("\nTraining model...")
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        self.evaluate_model(X_test, y_test_cat, y_test)
        
        # Save model and preprocessors
        self.save_model()
        
    def evaluate_model(self, X_test, y_test_cat, y_test):
        """Evaluate and visualize model performance"""
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.show()
        
        # Training history
        self.plot_training_history()
        
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
        
    def save_model(self):
        """Save model and preprocessors"""
        # Save model
        self.model.save(self.model_path)
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'input_shape': self.model.input_shape[1],
            'final_accuracy': float(self.history.history['val_accuracy'][-1]),
            'final_loss': float(self.history.history['val_loss'][-1])
        }
        
        with open('models/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\nModel saved to {self.model_path}")
        print(f"Final validation accuracy: {metadata['final_accuracy']:.4f}")

if __name__ == "__main__":
    trainer = GestureModelTrainer()
    trainer.train(epochs=100, batch_size=32)'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from datetime import datetime
import pickle

class GestureModelTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.history = None
        self.model_path = 'models/gesture_model.h5'
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        
    def load_data(self):
        """Load all collected data"""
        features_files = glob.glob('data/raw/features_*.npy')
        labels_files = glob.glob('data/raw/labels_*.npy')
        
        if not features_files:
            raise ValueError("No data files found! Please run extract_features.py first.")
        
        all_features = []
        all_labels = []
        
        for feat_file, label_file in zip(sorted(features_files), sorted(labels_files)):
            features = np.load(feat_file)
            labels = np.load(label_file)
            all_features.append(features)
            all_labels.extend(labels)
        
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        print(f"Loaded {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        print(f"Unique gestures: {np.unique(y)}")
        
        return X, y
    
    def augment_data(self, X, y):
        """Apply data augmentation"""
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Original
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Add noise
            noise = np.random.normal(0, 0.01, X[i].shape)
            augmented_X.append(X[i] + noise)
            augmented_y.append(y[i])
            
            # Scale slightly
            scale = np.random.uniform(0.95, 1.05)
            augmented_X.append(X[i] * scale)
            augmented_y.append(y[i])
            
            # Add more variations for better training
            # Slight rotation in feature space
            rotation_noise = np.random.normal(0, 0.005, X[i].shape)
            augmented_X.append(X[i] + rotation_noise)
            augmented_y.append(y[i])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def preprocess_data(self, X, y, test_size=0.2):
        """Preprocess and split data"""
        # Apply augmentation
        X_aug, y_aug = self.augment_data(X, y)
        print(f"After augmentation: {len(X_aug)} samples")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_aug)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_aug)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=42, stratify=y_encoded
        )
        
        # Convert to categorical for neural network
        num_classes = len(np.unique(y_encoded))
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {num_classes}")
        
        return X_train, X_test, y_train_cat, y_test_cat, y_train, y_test
    
    def build_advanced_model(self, input_shape, num_classes):
        """Build advanced neural network with attention mechanism"""
        inputs = keras.Input(shape=(input_shape,))
        
        # Feature extraction layers
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(256, activation='tanh')(x)
        attention = layers.Dense(256, activation='softmax')(attention)
        x = layers.Multiply()([x, attention])
        
        # Classification layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Use fixed learning rate instead of schedule
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), 
                    keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, epochs=100, batch_size=32):
        """Train the model"""
        # Load and preprocess data
        X, y = self.load_data()
        X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = self.preprocess_data(X, y)
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_advanced_model(X_train.shape[1], num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\nTraining model...")
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        self.evaluate_model(X_test, y_test_cat, y_test)
        
        # Save model and preprocessors
        self.save_model()
        
        return self.model, self.history
        
    def evaluate_model(self, X_test, y_test_cat, y_test):
        """Evaluate and visualize model performance"""
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Test accuracy
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        print("-" * 50)
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_mask = y_test == i
            class_correct = np.sum(y_pred[class_mask] == i)
            class_total = np.sum(class_mask)
            class_acc = class_correct / class_total if class_total > 0 else 0
            print(f"{class_name:20s}: {class_acc:.2%} ({class_correct}/{class_total})")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.show()
        
        # Training history
        self.plot_training_history()
        
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', color='blue')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', color='red')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', color='blue')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', color='red')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train', color='blue')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation', color='red')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train', color='blue')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation', color='red')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self):
        """Save model and preprocessors"""
        # Save model
        self.model.save(self.model_path)
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'input_shape': int(self.model.input_shape[1]),
            'final_accuracy': float(self.history.history['val_accuracy'][-1]),
            'final_loss': float(self.history.history['val_loss'][-1]),
            'final_precision': float(self.history.history['val_precision'][-1]) if 'val_precision' in self.history.history else 0,
            'final_recall': float(self.history.history['val_recall'][-1]) if 'val_recall' in self.history.history else 0,
            'total_epochs': len(self.history.history['loss']),
            'best_val_accuracy': float(max(self.history.history['val_accuracy']))
        }
        
        with open('models/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\nModel saved to {self.model_path}")
        print(f"Final validation accuracy: {metadata['final_accuracy']:.4f}")
        print(f"Best validation accuracy: {metadata['best_val_accuracy']:.4f}")
        print(f"Training completed in {metadata['total_epochs']} epochs")

if __name__ == "__main__":
    trainer = GestureModelTrainer()
    trainer.train(epochs=100, batch_size=32)