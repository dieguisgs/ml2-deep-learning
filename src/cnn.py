import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sys import platform
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import time
import seaborn as sns

from src.results import Results

if platform == "darwin":
    # Fix macOS error "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CNN:
    """Class to classify images using a transfer learning or fine-tuning on a pre-trained CNN.

        Examples:
            1. Training and evaluating the CNN. Optionally, save the model.
                cnn = CNN(base_model='ResNet50', model_name="ResNet50_prueba1)
                cnn.train(training_dir, validation_dir)
                cnn.predict(validation_dir)
                cnn.save(filename)

            2. Loading a trained CNN to evaluate against a previously unseen test set.
                cnn = CNN()
                cnn.load(filename)
                cnn.predict(test_dir)
    """

    def __init__(self,base_model: str = None, model_name:str = None, unfreezed_convolutional_layers: int = 0, include_top:bool = False):
        """CNN transfer learning class initializer."""
        self._base_model = base_model
        self._model_name = ""
        self._name_model_name = model_name
        self._model = None
        self._target_size = None
        self._preprocessing_function = None
        self.add_ = 0
        self._initialize_base_model(base_model, unfreezed_convolutional_layers, include_top=include_top)
        self._unfreezed_convolutional_layers = unfreezed_convolutional_layers
        

    def train(self, training_dir: str, validation_dir: str, epochs: int = 1, training_batch_size: int = 32, validation_batch_size: int = 32,
              learning_rate: float = 1e-4,early_stopping_patience: int = 3, reduce_lr_patience: int = None, reduce_lr_factor: float = None):
        """Use transfer learning or fine-tuning to train a base network to classify new categories.

        Args:
            training_dir: Relative path to the training directory (e.g., 'dataset/training').
            validation_dir: Relative path to the validation directory (e.g., 'dataset/validation').
            base_model: Pre-trained CNN { DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2, InceptionV3,
                                          MobileNet, MobileNetV2, NASNetLarge, NASNetMobile, ResNet50, VGG16, VGG19,
                                          Xception }.
            epochs: Number of times the entire dataset is passed forward and backward through the neural network.
            unfreezed_convolutional_layers: Starting from the end, number of trainable convolutional layers.
            training_batch_size: Number of training examples used in one iteration.
            validation_batch_size: Number of validation examples used in one iteration.
            learning_rate: Optimizer learning rate.

        """
        assert self._base_model is not None, "The base model must be specified"
        assert self._name_model_name is not None, "The model name must be specified"

        # Configure loading and pre-processing/data augmentation functions
        print('\n\nReading training and validation data...')
        training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self._preprocessing_function,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,  # Randomly flip half of the images horizontally
            fill_mode='nearest'  # Strategy used for filling in new pixels that appear after transforming images
        )
        #Aumenta los datos haciendo transformaciones que tengan sentido, por ejemplo imagenes con ligeras horientacones. Estirar, girar, zoom, etc
         # rotar la foto, esas tranformaicones provocna que el tamaño de la imagen cambie, por eso se usa fill_mode
         # Una técnica más razonable es copiar el pixel más próximo. Otra tecnica es la de copiar en espejo.
        #Se generan imagenes de entranamiento y validacion. En cada Batch pueda haber imagen de cualquier tipo. En el caso de la validación
        # como no se estan ajustando los pesos. No aplica el cambio de orden en el entrnamiento de las imegenes

        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self._preprocessing_function)

        training_generator = training_datagen.flow_from_directory(
            training_dir,
            target_size=self._target_size,
            batch_size=training_batch_size,
            class_mode='categorical'
        ) 

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self._target_size,
            batch_size=validation_batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Add a new softmax output layer to learn the training dataset classes
        if self.add_ == 0:
            self._add_output_layers(training_generator.num_classes)
        else:
            assert training_generator.num_classes == self._model.output_shape[1], f"The number of classes in the training set must be the same as the number of classes in the model output layer. The num of classes in the training set is {training_generator.num_classes}"

    

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self._model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Display a summary of the model
        print('\n\nModel summary')
        self._model.summary() #El rsumen de las capas sale aquí

        # Callbacks. Check https://www.tensorflow.org/api_docs/python/tf/keras/callbacks for more alternatives.
        # EarlyStopping and ModelCheckpoint are probably the most relevant. Se hace para que el modelo no sobreaprenda.
        #Con pocas epocas de entrenamiento no hace falta hacer un early stopping pero si tengo muchas igual si.

        # To launch TensorBoard type the following in a Terminal window: tensorboard --logdir /path/to/log/folder
        if not os.path.exists(os.path.abspath(f"./src/cnns/{self._name_model_name}/logs")):
            os.makedirs(os.path.abspath(f"./src/cnns/{self._name_model_name}/logs"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.abspath(f"./src/cnns/{self._name_model_name}/logs"), histogram_freq=0,
            write_graph=True, write_grads=False,
            write_images=False, embeddings_freq=0,
            embeddings_layer_names=None, embeddings_metadata=None,
            embeddings_data=None, update_freq='epoch'
        ) 

        earlystopping = EarlyStopping(monitor='val_loss',
                                             min_delta = 0, 
                                             patience = early_stopping_patience, #Si no mejora en 3 epocas para
                                             verbose = 0,
                                             restore_best_weights = True #devuelve la primera epoca a partir de la cual no mejoro en el entrenamiento
        )
        if reduce_lr_patience is not None and reduce_lr_factor is not None:
            reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, verbose=0, min_delta=0.0001)
            callbacks = [tensorboard_callback, earlystopping,reduce_on_plateau]
        else:
            callbacks = [tensorboard_callback, earlystopping]
        if reduce_lr_factor is None and reduce_lr_patience is  None:
            self._reduce_lr = None
        else:
            self._reduce_lr = {'monitor': reduce_on_plateau.monitor, 'patience': reduce_on_plateau.patience, 'factor': reduce_on_plateau.factor, 'min_delta': reduce_on_plateau.min_delta}
        self._earlystopping = {'monitor': earlystopping.monitor , 'patience': earlystopping.patience, 'min_delta': earlystopping.min_delta, 'restore_best_weights': earlystopping.restore_best_weights}
        self._batch_size = {"Train": training_batch_size, "Validation": validation_batch_size}
        self._learning_rate = learning_rate
        self._epochs = epochs

    

        #Con tensorboard se guardan logs de los entrenamientos.

        # Train the network
        print("\n\nTraining CNN...")
        #time the history of the training
        start = time.time()
        history = self._model.fit(
            training_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        end = time.time()
        self._training_time = end - start
        self._history = history


        self._dropout_rates = []
        self._activation_denses = []
        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                self._dropout_rates.append(layer.rate)
            if isinstance(layer, tf.keras.layers.Dense):
                self._activation_denses.append((str(layer.activation).split(' ')[1], layer.output.shape[1]))

        # Plot model training history
        if epochs > 1:
            self._fig=self._plot_training(history)

    def predict(self, test_dir: str, dataset_name: str = "", save: bool = True):
        """Evaluates a new set of images using the trained CNN.

        Args:
            test_dir: Relative path to the validation directory (e.g., 'dataset/test').
            dataset_name: Dataset descriptive name.
            save: Save results to an Excel file.

        """
        # Configure loading and pre-processing functions
        print('Reading test data...')
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self._preprocessing_function)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self._target_size,
            batch_size=1,  # A batch size of 1 ensures that all test images are processed
            class_mode='categorical',
            shuffle=False
        )

        # Predict categories
        predictions = self._model.predict(test_generator)
        predicted_labels = np.argmax(predictions, axis=1).ravel().tolist()

        # Format results and compute classification statistics
        results = Results(test_generator.class_indices, dataset_name=dataset_name, model_name=self._name_model_name)
        accuracy, confusion_matrix, classification = results.compute(test_generator.filenames, test_generator.classes,
                                                                     predicted_labels)
        # Display and save results
        results.print(accuracy, confusion_matrix)

        if save:
            results.save(confusion_matrix, classification, predictions)
            if os.path.isfile(f"./src/cnns/{self._name_model_name}/validation_results.xlsx") and os.path.isfile(f"./src/cnns/{self._name_model_name}/training_results.xlsx"):
                accuracy_val = self._accuracy_excel(f"./src/cnns/{self._name_model_name}/validation_results.xlsx")
                accuracy_train = self._accuracy_excel(f"./src/cnns/{self._name_model_name}/training_results.xlsx")
                if not os.path.isfile('src/cnns/overall_results.xlsx'):
                    overall_df = pd.DataFrame(columns=['Model', 'Base Model', 'Accuracy Train', 'Accuracy Val', 'Nº Last Layers', 'Last Layers', 'DropOut', 'Dense Activations', 'Epochs', 'Learning Rate', 'EarlyStopping',
                                                        'Batch', 'Nº Params', 'Unfreezed CNN Layers', 'Elapsed Time'])
                    overall_df.to_excel('src/cnns/overall_results.xlsx', index=False)
                overall_df = pd.read_excel('src/cnns/overall_results.xlsx')
                new_row = pd.DataFrame({'Model': self._name_model_name, "Base Model": self._base_model, 'Accuracy Train': accuracy_train,
                                        'Accuracy Val': accuracy_val, 'Nº Last Layers': len(self._model.layers), 'Last Layers': str([str(type(layer)).split(".")[-1][:-2] for layer in self._model.layers]),
                                        "DropOut": str(self._dropout_rates), "Dense Activations": str(self._activation_denses), 'Epochs': self._epochs, "Learning Rate": self._learning_rate, 'EarlyStopping': str(self._earlystopping),
                                        "Batch": str(self._batch_size), "Nº Params": self._model.count_params(), "Unfreezed CNN Layers": self._unfreezed_convolutional_layers,
                                        "Elapsed Time": self._training_time}, index=[0])
                overall_df = pd.concat([overall_df, new_row], ignore_index=True)
                overall_df.to_excel('src/cnns/overall_results.xlsx', index=False)

    
    def load(self,filename: str):
        """Loads a trained CNN model and the corresponding preprocessing information.

        Args:
           filename: Relative path to the file without the extension.

        """
        # Load Keras model
        self._model = tf.keras.models.load_model(f'src/cnns/{filename}/'+filename + '.h5')

        # Load base model information
        with open(f'src/cnns/{filename}/'+filename + '.json') as f:
            self._model_name = json.load(f)

        self._initialize_attributes()
    def add(self,tf_keras_layers):
        """
        Add a new layer to the model like 
        cnn.add(tf.keras.layers.Dense(1024/2, activation='relu'))
        cnn.add(tf.keras.layers.Dropout(0.5))
        cnn.add(tf.keras.layers.Dense(15, activation='softmax'))
        Args:
            tf_keras_layers: A layer of tensorflow keras
        """
                # Create a new model
        
        if self.add_ == 0:
            self._model2 = tf.keras.models.Sequential()
            # Add the convolutional base model
            self._model2.add(self._model)
            
        else:
            self._model2.add(tf_keras_layers)
        self.add_ += 1
        self._model = self._model2

    def save(self,):
        """Saves the model to an .h5 file and the model name to a .json file.

        Args:
           filename: Relative path to the file without the extension.

        """
        #if  folder src/cnn_results/ does not exist, create it
        if not os.path.exists(f'src/cnns/{self._name_model_name}'):
            os.makedirs(f'src/cnns/{self._name_model_name}')


        # Save Keras model
        self._model.save(f'src/cnns/{self._name_model_name}/'+self._name_model_name + '.h5')

        # Save base model information
        with open(f'src/cnns/{self._name_model_name}/'+self._name_model_name + '.json', 'w', encoding='utf-8') as f:
            json.dump(self._model_name, f, ensure_ascii=False, indent=4, sort_keys=True)

        #save the image of _plot_training

        self._fig.savefig(f'src/cnns/{self._name_model_name}/'+self._name_model_name + '.png')


    def _initialize_base_model(self, base_model: str, unfreezed_convolutional_layers: int, include_top: bool = True,
                               pooling: str = 'avg'):
        """Initializes the base model.

        Args:
            base_model: Pre-trained CNN { DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2, InceptionV3,
                                          MobileNet, MobileNetV2, NASNetLarge, NASNetMobile, ResNet50, VGG16, VGG19,
                                          Xception }.
            unfreezed_convolutional_layers: Starting from the end, number of trainable convolutional layers.
            include_top: True to use the full base model; false to remove the last classification layers.
            pooling: Optional pooling mode for feature extraction when include_top is False
                - None: The output of the model will be the 4D tensor output of the last convolutional block.
                - 'avg': Global average pooling will be applied to the output of the last convolutional block, and thus
                         the output of the model will be a 2D tensor.
                - 'max': Global max pooling will be applied.

        Raises:
            TypeError: If the unfreezed_convolutional_layers parameter is not an integer.
            ValueError: If the unfreezed_convolutional_layers parameter is not a positive number (>= 0).
            ValueError: If the base model is not known.

        """
        self._model_name = base_model
        self._initialize_attributes()

        input_shape = self._target_size + (3,)

        # Initialize the base model. Loads the network weights from disk.
        # NOTE: If this is the first time you run this function, the weights will be downloaded from the Internet.
        self._model = getattr(tf.keras.applications, base_model)(weights='imagenet', include_top=include_top,
                                                              input_shape=input_shape, pooling=pooling)

        # Freeze convolutional layers
        if type(unfreezed_convolutional_layers) != int:
            raise TypeError("unfreezed_convolutional_layers must be a positive integer.")

        if unfreezed_convolutional_layers == 0:
            freezed_layers = self._model.layers
        elif unfreezed_convolutional_layers > 0:
            freezed_layers = self._model.layers[:-unfreezed_convolutional_layers]
        else:
            raise ValueError("unfreezed_convolutional_layers must be a positive integer.")

        for layer in freezed_layers:
            layer.trainable = False

    def _initialize_attributes(self):
        """Initialize the input image shape along with the pre-processing function.

        Raises:
            ValueError: If the model is unknown.

        """
        if self._model_name in ('DenseNet121', 'DenseNet169', 'DenseNet201'):
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.densenet.preprocess_input
        elif self._model_name == 'InceptionResNetV2':
            self._target_size = (299, 299)
            self._preprocessing_function = tf.keras.applications.inception_resnet_v2.preprocess_input
        elif self._model_name == 'InceptionV3':
            self._target_size = (299, 299)
            self._preprocessing_function = tf.keras.applications.inception_v3.preprocess_input
        elif self._model_name == 'MobileNet':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
        elif self._model_name == 'MobileNetV2':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
        elif self._model_name == 'NASNetLarge':
            self._target_size = (331, 331)
            self._preprocessing_function = tf.keras.applications.nasnet.preprocess_input
        elif self._model_name == 'NASNetMobile':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.nasnet.preprocess_input
        elif self._model_name == 'ResNet50':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        elif self._model_name == 'VGG16':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.vgg16.preprocess_input
        elif self._model_name == 'VGG19':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif self._model_name == 'Xception':
            self._target_size = (299, 299)
            self._preprocessing_function = tf.keras.applications.xception.preprocess_input
        else:
            raise ValueError("Base model not supported. Possible values are 'DenseNet121', 'DenseNet169', "
                             "'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', "
                             "'NASNetLarge', 'NASNetMobile', 'ResNet50', 'VGG16', 'VGG19' and 'Xception'.")
    # Una forma de evitar que la red sobreapenda es ponerle dificil las cosas en el entrenamiento. El DropOut hace que que el x% de las conexiones no
    #se puedan utilzar en cada iteracion. Esto hace que la red tenga que aprender de otras conexiones y no se sobreapenda.
    def _add_output_layers(self, class_count: int, fc_layer_size: int = 1024/2):
        """Append a fully-connected shallow neural network with softmax outputs at the end of the base model.

        Args:
          class_count: Number of classes (i.e., number of output softmax neurons).
          fc_layer_size: Number of neurons in the hidden layer.

        """
        # Create a new model
        model = tf.keras.models.Sequential()

        # Add the convolutional base model
        model.add(self._model)

        # Add new layers
        model.add(tf.keras.layers.Dense(fc_layer_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.15)) #Dropout seguramente haya que modificarlo
        model.add(tf.keras.layers.Dense(fc_layer_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.15)) #Dropout seguramente haya que modificarlo
        model.add(tf.keras.layers.Dense(class_count, activation='softmax'))

        # Assign the new model to the class attribute
        self._model = model

    @staticmethod
    def _plot_training(history):
        """Plots the evolution of the accuracy and the loss of both the training and validation sets.

        Args:
            history: Training history.

        """
        training_accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        sns.set_style("darkgrid")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].set_ylabel('Loss', fontsize=16)
        axs[0].plot(loss, label='Training Loss', color='tab:blue', linewidth=1)
        axs[0].plot(val_loss, label='Validation Loss', color='tab:orange', linewidth=1)
        axs[0].legend(loc='upper right')

        axs[1].set_ylabel('Accuracy', fontsize=16)
        axs[1].plot(training_accuracy, label='Training Accuracy', color='tab:blue', linewidth=1)
        axs[1].plot(validation_accuracy, label='Validation Accuracy', color='tab:orange', linewidth=1)
        axs[1].legend(loc='lower right')
        fig.suptitle('CNN Loss/Accuracy', fontsize=16)
        return fig

    def _accuracy_excel(self,excel_file_name:str):
        confusion_matrix = pd.read_excel(excel_file_name, index_col=0)

        correct_predictions = confusion_matrix.values.diagonal().sum()
        total_predictions = confusion_matrix.values.sum()

        accuracy = correct_predictions / total_predictions

        return  accuracy
