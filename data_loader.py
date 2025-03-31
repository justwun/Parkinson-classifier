import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# size for images (don't change unless u know what ur doing)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 
# data augmentation for training, makes images funky so model learns better
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10, zoom_range=0.1)
# validation data (no funky stuff just rescale)
val_datagen = ImageDataGenerator(rescale=1./255)
data_dir = "C:\Users\VINH\Desktop\SHAKYSHAKY\DATA\merged"
train_generator = train_datagen.flow_from_directory(directory=os.path.join(data_dir, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
val_generator = val_datagen.flow_from_directory(directory=os.path.join(data_dir, "val"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
