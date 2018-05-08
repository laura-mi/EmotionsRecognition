from keras.preprocessing.image import ImageDataGenerator      
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import pandas as pd

from Services.DatasetService import getFerDataset
from Services.DatasetService import splitData
from Services.ImageProcessingService import preprocessForFaces
from Networks.CnnEmotionClassifier import getModel

# parameters
input_shape = (64, 64, 1)

# data generator
data_generator = ImageDataGenerator(featurewise_center=False,featurewise_std_normalization=False,rotation_range=10,
                        width_shift_range=0.1,height_shift_range=0.1,zoom_range=.1,horizontal_flip=True)

# model parameters
model = getModel(input_shape)

#model.summary()

# callbacks
def getCallbacks(patience):
    
    #define paths
    base_path = '../TrainedModels/Emotions/'
    dataset_name = 'fer2013'
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    
    #Callback that streams epoch results to a csv file.
    csv_logger = CSVLogger(log_file_path, append=False)

    #Stop training when a monitored quantity has stopped improving.
    #monitor: quantity to be monitored.
    #patience: number of epochs with no improvement after which training will be stopped.
    early_stop = EarlyStopping('val_loss', patience=patience)

    #Reduce learning rate when a metric has stopped improving.
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,patience=int(patience/4), verbose=1)

    #Save the model after every epoch.
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)

    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
    return callbacks


# parameters
batch_size = 64
num_epochs = 256
validation_split = .2
patience = 50
callbacks = getCallbacks(patience)

# loading dataset
faces, emotions = getFerDataset(input_shape[:2])
faces = preprocessForFaces(faces)
num_samples, num_classes = emotions.shape
train_data, val_data = splitData(faces, emotions, validation_split)
train_faces, train_emotions = train_data

#Trains the model on data generated batch-by-batch by a Python generator or an instance of Sequence.
#The generator is run in parallel to the model, for efficiency.
#For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.
#Todo: use_multiprocessing=True
model.fit_generator(data_generator.flow(train_faces, train_emotions,batch_size),steps_per_epoch=len(train_faces) / batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks,validation_data=val_data)
#model = emotionClassifier