from data_loader import train_generator, val_generator
from model_definition import model
history = model.fit(
    train_generator,  
    validation_data=val_generator,  
    epochs=60 
)
