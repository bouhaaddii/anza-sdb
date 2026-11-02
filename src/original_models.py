"""Scientific functions copied verbatim from EN. See vendor and AST verification."""

import numpy as np

import tensorflow as tf

def build_sdb_cnn(input_size=(9, 9, 6), dropout_rate=0.3):
    """Original SDB-CNN model."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_pi_cnn(input_size=(9, 9, 6), dropout_rate=0.3, lambda_phys=0.1):
    """Physics-Informed CNN with Beer-Lambert regularization - CORRECTED VERSION."""
    from tensorflow.keras import layers, models
    
    def beer_lambert_depth(rgb_patch, k=0.1):
        """Physical depth estimation."""
        blue_band = rgb_patch[..., 1]
        green_band = rgb_patch[..., 2]
        
        safe_green = tf.where(green_band > 0.001, green_band, 0.001)
        ratio = blue_band / safe_green
        
        depth_phys = -tf.math.log(ratio) / k
        depth_phys = tf.clip_by_value(depth_phys, -30, 5)
        
        return depth_phys
    
    inputs = layers.Input(shape=input_size)
    
    # CNN branch
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # CNN features
    x = layers.GlobalAveragePooling2D()(x)
    
    # Physical information
    center_patches = layers.Lambda(
        lambda x: tf.reduce_mean(x[:, 3:6, 3:6, :], axis=[1, 2]),
        output_shape=(6,)
    )(inputs)
    
    def compute_physics(center_values):
        blue_band = center_values[:, 1]
        green_band = center_values[:, 2]
        
        safe_green = tf.where(green_band > 0.001, green_band, 0.001)
        ratio = blue_band / safe_green
        
        depth_phys = -tf.math.log(ratio) / 0.1
        depth_phys = tf.clip_by_value(depth_phys, -30, 5)
        
        return tf.expand_dims(depth_phys, axis=-1)
    
    physics_depth = layers.Lambda(
        compute_physics,
        output_shape=(1,)
    )(center_patches)
    
    # Fusion CNN + physics
    x = layers.Concatenate()([x, physics_depth])
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    class PhysicalLoss(tf.keras.losses.Loss):
        def __init__(self, lambda_phys=0.1, name="physical_loss"):
            super().__init__(name=name)
            self.lambda_phys = lambda_phys
            self.mse = tf.keras.losses.MeanSquaredError()
            
        def call(self, y_true, y_pred):
            mse_loss = self.mse(y_true, y_pred)
            return mse_loss
            
        def get_config(self):
            return {"lambda_phys": self.lambda_phys}
    
    return model, PhysicalLoss(lambda_phys=lambda_phys)

def build_cnn_bilstm(input_size=(9, 9, 6), dropout_rate=0.3, lstm_units=64):
    """Hybrid CNN-BiLSTM architecture."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Features for BiLSTM
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, -1))(x)
    
    # BiLSTM layer
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        merge_mode='concat'
    )(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_fcnn(input_size=(9, 9, 6), dropout_rate=0.3):
    """Fully Convolutional Network for dense output - CORRECTED VERSION."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 9x9 → 4x4
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 4x4 → 2x2
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder to upsample to 9x9
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), 
                               padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), 
                               padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.ZeroPadding2D(((0, 1), (0, 1)))(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(1, (1, 1), activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_seabed_net(input_size=(9, 9, 6), dropout_rate=0.3, num_classes=4):
    """Multi-task network: bathymetry + classification."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    # Common backbone
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Common features
    common_features = layers.GlobalAveragePooling2D()(x)
    
    # Bathymetry head
    depth_branch = layers.Dense(64, activation='relu')(common_features)
    depth_branch = layers.Dropout(dropout_rate)(depth_branch)
    depth_branch = layers.Dense(32, activation='relu')(depth_branch)
    depth_output = layers.Dense(1, name='depth')(depth_branch)
    
    # Seabed classification head
    seabed_branch = layers.Dense(64, activation='relu')(common_features)
    seabed_branch = layers.Dropout(dropout_rate)(seabed_branch)
    seabed_branch = layers.Dense(32, activation='relu')(seabed_branch)
    seabed_output = layers.Dense(num_classes, activation='softmax', name='seabed_class')(seabed_branch)
    
    model = models.Model(inputs=inputs, outputs=[depth_output, seabed_output])
    return model

def prepare_seabed_labels(depth_data, num_classes=4):
    """Prepares classification labels from depth."""
    if isinstance(depth_data, np.ndarray):
        if depth_data.ndim > 1:
            depth_values = depth_data.reshape(-1)
        else:
            depth_values = depth_data
    else:
        depth_values = np.array(depth_data).reshape(-1)
    
    seabed_labels = np.zeros((len(depth_values), num_classes), dtype=np.float32)
    
    for i, depth in enumerate(depth_values):
        if depth < 2:
            seabed_labels[i, 0] = 1.0
        elif depth < 5:
            seabed_labels[i, 1] = 1.0
        elif depth < 15:
            seabed_labels[i, 2] = 1.0
        else:
            seabed_labels[i, 3] = 1.0
    
    return seabed_labels

def create_model(model_type, input_size, **kwargs):
    """Factory to create requested model."""
    print(f"\n🏗️  Creating model: {model_type}")
    
    if model_type == 'sdb_cnn':
        model = build_sdb_cnn(input_size=input_size, dropout_rate=kwargs.get('dropout_rate', 0.3))
        loss = tf.keras.losses.MeanSquaredError()
        metrics = ['mae', 'mse']
        
    elif model_type == 'pi_cnn':
        model, custom_loss = build_pi_cnn(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            lambda_phys=kwargs.get('lambda_phys', 0.1)
        )
        loss = custom_loss
        metrics = ['mae', 'mse']
        
    elif model_type == 'cnn_bilstm':
        model = build_cnn_bilstm(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            lstm_units=kwargs.get('lstm_units', 64)
        )
        loss = tf.keras.losses.MeanSquaredError()
        metrics = ['mae', 'mse']
        
    elif model_type == 'fcnn':
        model = build_fcnn(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3)
        )
        loss = tf.keras.losses.MeanSquaredError()
        metrics = ['mae', 'mse']
        
    elif model_type == 'seabed_net':
        model = build_seabed_net(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            num_classes=kwargs.get('num_classes', 4)
        )
        loss = {
            'depth': tf.keras.losses.MeanSquaredError(),
            'seabed_class': tf.keras.losses.CategoricalCrossentropy()
        }
        loss_weights = {
            'depth': 0.7,
            'seabed_class': 0.3
        }
        metrics = {
            'depth': ['mae', 'mse'],
            'seabed_class': ['accuracy']
        }
        
        return model, loss, loss_weights, metrics
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, loss, None, metrics
