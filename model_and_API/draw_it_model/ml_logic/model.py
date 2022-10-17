from colorama import Fore, Style
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping


def initialize_model():
    """
    Initialize the Neural Network
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)


    #Model architecture
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape = (64,64,3))
    base_model = set_nontrainable_layers(vgg_model)

    model = Sequential([
        Rescaling(1./255),
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(345, activation='softmax')
        ])

    print("\n✅ model initialized")

    return model


def set_nontrainable_layers(model):
    '''
    Sets model layers to trainable or nontrainable
    '''
    #Nontrainable layers
    for layer in model.layers[:11]:
        layer.trainable = False

    #Trainable layers
    for layer in model.layers[11:]:
        layer.trainable = True

    #Confirmation that the layers are trainable or nontrainable
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    return model


def compile_model(model, learning_rate):
    """
    Compile the Neural Network
    """

    #Define optimizer and set learning_rate
    opt = optimizers.Adam(learning_rate=learning_rate)

    #Compile model - to be updated based on approach (e.g., Categorical Crossentropy)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    print("\n✅ model compiled")

    return model


def train_model(model,
                train_ds,
                val_ds,
                patience=2):
    """
    Fit model and return the fitted model and history
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    #Training setup - to be adjusted (e.g., EarlyStopping definition, number of epochs etc.)
    es = EarlyStopping(monitor = 'val_accuracy',
                       mode = 'max',
                       patience = patience,
                       verbose = 1,
                       restore_best_weights = True)

    #Train model
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs = 20,
                        callbacks=[es],
                        verbose = 1
                        )

    print(f"\n✅ model trained")

    return model, history


def evaluate_model(model,
                   dataset,
                   batch_size=64):
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model test data" + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    #Evaluate model - to be adjusted based on approach
    metrics = model.evaluate(
        dataset,
        verbose=1,
        return_dict=True
    )

    #Store loss and accuracy
    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} accuracy {round(accuracy, 2)}")

    return metrics
