from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.optimizers import Adam

def build_model(num_classes):
    base_model=EfficientNetB0(include_top=False,weights='imagenet',input_shape=(224,224,3))
    base_model.trainable=False

    x=GlobalAveragePooling2D()(base_model.output)
    x=Dropout(0.5)(x)
    output=Dense(num_classes,activation='softmax')(x)

    model=Model(inputs=base_model.input,outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

    return model