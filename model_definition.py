from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  
model = models.Sequential([
    base_model, 
    layers.GlobalAveragePooling2D(),  
    layers.Dense(128, activation='relu'), 
    layers.Dropout(0.5),  
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary() 
