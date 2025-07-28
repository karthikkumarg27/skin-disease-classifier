import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(image_dir,csv_path):
    df=pd.read_csv(csv_path)

    df['image_id']=df['image_id'].apply(lambda x:x+".jpg")

    label_map={
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }

    df['label']=df['dx'].map(label_map)

    datagen=ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2
    )

    train_gen=datagen.flow_from_dataframe(
        df,
        directory=image_dir,
        x_col="image_id",
        y_col="label",
        subset="training",
        target_size=(224,224),
        class_mode="categorical",
        batch_size=32
    )

    val_gen=datagen.flow_from_dataframe(
        df,
        directory=image_dir,
        x_col="image_id",
        y_col="label",
        subset="validation",
        target_size=(224,224),
        class_mode="categorical",
        batch_size=32
    )

    return train_gen,val_gen