# Common
import mlflow
import mlflow.tensorflow
from difflib import SequenceMatcher
import numpy as np
import tensorflow as tf
from IPython.display import clear_output as cls
# Data
from glob import glob
from tqdm import tqdm
import tensorflow.data as tfd
# Model
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd 

mlflow.start_run(run_name="OCR-Experiment")

pdf = pd.read_csv('images_data_v1\labels.csv')
# images_data_v1\images\1-1.png
dirp = 'images_data_v1/images/'
pdf['FILENAME'] = [dirp + filename.split('/')[-1] for filename in pdf['FILENAME']]

merged_df=pdf[:10]

train_csv , valid_csv =  train_test_split(merged_df ,test_size=0.05, random_state=42)




def load_image(image_path: str):
    # Read the Image
    image = tf.io.read_file(image_path)

    # Decode the image
    decoded_image = tf.image.decode_png(contents=image, channels=1)

    # Convert image data type.
    cnvt_image = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)

    # Resize the image
    resized_image = tf.image.resize(images=cnvt_image, size=(IMG_HEIGHT, IMG_WIDTH))

    # Flip image
    resized_image = tf.image.flip_left_right(resized_image)

    # Transpose
    image = tf.transpose(resized_image, perm=[1, 0, 2])

    # Convert image to a tensor.
    image = tf.cast(image, dtype=tf.float32)

    # Return loaded image
    return image


# Image Size
IMG_WIDTH = 2300
IMG_HEIGHT = 250
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Batch Size
BATCH_SIZE = 3

# EPOCHS
EPOCHS = 1

# Model Name
MODEL_NAME = 'Handwritten-OCR'

# Callbacks
CALLBACKS = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath=MODEL_NAME + ".h5", save_best_only=True)
]

# Learning Rate
LEARNING_RATE = 1e-3

# Random Seed
np.random.seed(2569)
tf.random.set_seed(2569)

# AUTOTUNE
AUTOTUNE = tfd.AUTOTUNE

# unique_chars= ['ل', 'پ', 'و', 'م', 'چ', 'ر', 'ب', 'ا', 'ہ', 'گ', 'ع', 'ج', 'آ', 'ڑ', 'خ', 'ظ', 'ھ', 'غ', 'ص', 'ے', 'ح', 'ں', 'ٹ', 'ؤ', 'ڈ', 'ش', 'س', 'ف', 'ک', 'د', 'ط', ' ', 'ض', 'ئ', 'ز', 'ق', 'ت', 'ی', 'ذ', 'ن']

train_labels = [str(word) for word in merged_df['IDENTITY'].to_numpy()]
train_labels[:10]
# Unique characters
unique_chars = set(char for word in train_labels for char in word)
n_classes = len(unique_chars)

# Show
print(f"Total number of unique characters : {n_classes}")
print(f"Unique Characters : \n{unique_chars}")

char_to_num = layers.StringLookup(
    vocabulary = list(unique_chars),
    mask_token = None
)

# Reverse dictionary
num_to_char = layers.StringLookup(
    vocabulary = char_to_num.get_vocabulary(),
    mask_token = None, 
    invert = True
)
MAX_LABEL_LENGTH = max(map(len, train_labels))
print(f"Maximum length of a label : {MAX_LABEL_LENGTH}")


def encode_single_sample(image_path : str, label : str):
    # Get the image
    image = load_image(image_path)
    
    # Convert the label into characters
    chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
    
    # Convert the characters into vectors
    vecs = char_to_num(chars)
    
    # Pad label
    pad_size = MAX_LABEL_LENGTH - tf.shape(vecs)[0]
    vecs = tf.pad(vecs, paddings = [[0, pad_size]], constant_values=n_classes+1)
    
    return {'image':image, 'label':vecs}

# Training Data
train_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(train_csv['FILENAME'].to_list()), np.array(train_csv['IDENTITY'].to_list()))
).shuffle(1000).map(encode_single_sample).batch(BATCH_SIZE)

# Validation data
valid_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(valid_csv['FILENAME'].to_list()), np.array(valid_csv['IDENTITY'].to_list()))
).map(encode_single_sample).batch(BATCH_SIZE)

class CTCLayer(layers.Layer):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.loss_fn = keras.backend.ctc_batch_cost
    
    def call(self, y_true, y_pred):
        
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        
        input_len = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_len = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        
        loss = self.loss_fn(y_true, y_pred, input_len, label_len)
        
        self.add_loss(loss)
        
        return y_pred



def create_model():

    # Input Layer
    input_images = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image")

    # Labels : These are added for the training purpose.
    target_labels = layers.Input(shape=(None, ), name="label")

    # CNN Network
    x = layers.Conv2D(
        filters=32, 
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(input_images)

    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    x = layers.Conv2D(
        filters=64, 
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal'
    )(x)

    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    # Encoding Space
    new_dim = (IMG_HEIGHT//4)*64
    encoding = layers.Reshape(target_shape=((IMG_WIDTH//4), new_dim))(x)
    encoding = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(encoding)
    encoding = layers.Dropout(0.2)(encoding)

    # RNN Network
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(encoding)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output Layer
    output = layers.Dense(len(char_to_num.get_vocabulary())+1, activation='softmax')(x)

    # CTC Layer
    ctc_layer = CTCLayer()(target_labels, output)

    # Model 
    model = keras.Model(
        inputs=[input_images, target_labels],
        outputs=[ctc_layer]
    )

    return model


def decode_pred(pred_label):
    
    # Input length
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
    
    # CTC decode
    decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][:,:MAX_LABEL_LENGTH]
    # Converting numerics back to their character values
    chars = num_to_char(decode)
    
    # Join all the characters
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
    
    # Remove the unknown token
    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]
    print(texts)

    return filtered_texts

# Define a function to calculate the CER between two lists of strings
def cer(list1, list2):
    total_dist = 0
    total_len = 0
    for i in range(len(list1)):
        s = SequenceMatcher(None, list1[i], list2[i])
        total_dist += (1 - s.ratio()) * len(list1[i])
        total_len += len(list1[i])
    return total_dist / total_len

mlflow.set_experiment("OCR Experiment")

mlflow.end_run()
# Start Run
with mlflow.start_run():
    # Log Parameters
    mlflow.log_param("IMG_WIDTH", IMG_WIDTH)
    mlflow.log_param("IMG_HEIGHT", IMG_HEIGHT)
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("EPOCHS", EPOCHS)
    mlflow.log_param("LEARNING_RATE", LEARNING_RATE)

    # Create and compile the model
    model = create_model()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer)
    
    # Fit the model
    history = model.fit(
        train_ds, 
        validation_data=valid_ds, 
        epochs=EPOCHS,
    )
    

    # Model required for inference
    inference_model = keras.Model(
        inputs=model.get_layer(name="image").input,
        outputs=model.get_layer(name='dense_1').output
    )

    # Model summary
    inference_model.summary()

    data = next(iter(valid_ds))
    images, labels = data['image'], data['label']
    predli=[]
    actualli=[]
    # Iterate over the data 
    for index, (image, label) in enumerate(zip(images, labels)):

        # Label processing
        text_label = num_to_char(label)
        text_label = tf.strings.reduce_join(text_label).numpy().decode('UTF-8')
        text_label = text_label.replace("[UNK]", "").strip()
        pred = inference_model.predict(tf.expand_dims(image, axis=0))
        pred = decode_pred(pred)[0]
        predli.append(pred)
        actualli.append(text_label)

    cer_val = cer(actualli, predli)
    print(f"CER: {cer_val}")

    # Log Metrics
    mlflow.log_metric("CER", cer_val)

    saved_model_path='C:/Users/Majid Ahmad/Desktop/dvc/model_store'
    model_name='Handwritten-OCR'
    mlflow.tensorflow.log_model(
            model=model,  # This should be your TensorFlow model object
            artifact_path='model',  # The run-relative artifact path
            registered_model_name='Handwritten-OCR'  # If you want to register the model as well
        )

# End Run
mlflow.end_run()
