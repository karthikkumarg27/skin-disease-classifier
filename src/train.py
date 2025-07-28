from model import build_model
from dataloader import load_data
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import os

def train_model():
    image_dir="data/HAM10000_images"
    csv_path="data/HAM10000_metadata.csv"

    train_gen,val_gen=load_data(image_dir,csv_path)
    model = build_model(num_classes=len(train_gen.class_indices))


    callbacks=[
        EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
        ModelCheckpoint("best_model.h5",save_best_only=True)
    
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=callbacks
    )

    model.save("final_model.h5")

if __name__ =="__main__":
    train_model()

