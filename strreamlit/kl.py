import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import io
import tensorflow as tf

# your characters
unique_chars = ['٢', 'ه', '6', '”', '؛', 'ؤ', '9', 'ی', 'ٴ', '٧', 'A', 'أ', 'ٰ', 's', '!', 'ہ', '٥', '#', '8', 'ﷲ', '«', '5', ';', '4', 'E', '۱', '٩', 'ّ', 'ً', 'َ', 'ک', '\ufeff', 'ئ', 'u', '‘', 'ث', 'ق', 'د', '․', '؟', 'ؐ', '١', 'O', '"', 'ٓ', '۳', '٬', '\n', 'ش', '؍', '[', '۴', "'", '۶', '۹', 'ل', ']', '“', 'ؓ', '’', 'آ', 'ٹ', 'ۃ', '۲', 'ژ', 'ؑ', 'ر', 'س', '(', ' ', 'ھ', 'ۓ', 'ن', 'ؒ', '٨', '_', 'ے', 'ح', '2', 'ء', 'و', 'ظ', 'M', 'ض', 'ﺅ', 'ْ', '\t', 'غ', 'ا', '.', 'م', '٠', '7', 'ذ', '*', 'ع', '،', 'ز', 'ڈ', ')', 'ف', 'ج', '٤', '۸', ',', '۔', '۵', '/', 'L', '\u200c', 'ي', '0', 'ڑ', '۷', ':', 'D', '3', 'پ', '…', 'ٔ', 'ٗ', '¿', 'ط', '-', '\u200f', 'ں', 'ُ', 'ص', 'ب', '٣', 'ۂ', 'گ', 'ت', '1', 'خ', 'چ', '۰', '٦', 'ِ']
n_classes = len(unique_chars)

# mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(unique_chars),
    mask_token=None,
)

# mapping integers to characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True,
)

MAX_LABEL_LENGTH = 99


def load_image(image):
    # Convert the PIL Image object to a NumPy array as our OCR function expects that.
    image_tensor = np.array(image)

    if image_tensor.shape[-1] != 1:
        image_tensor = tf.image.rgb_to_grayscale(image_tensor)

    cnvt_image = tf.image.convert_image_dtype(image=image_tensor, dtype=tf.float32)
    resized_image = tf.image.resize(images=cnvt_image, size=(250, 2300))
    resized_image = tf.image.flip_left_right(resized_image)
    image = tf.transpose(resized_image, perm=[1, 0, 2])
    image = tf.cast(image, dtype=tf.float32)

    return image


def decode_pred(pred_label):
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
    decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][:, :MAX_LABEL_LENGTH]
    chars = num_to_char(decode)
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]
    return filtered_texts


# loading the model
inference_model = keras.models.load_model('UrduOCRCheckpoint')


def Urdu_OCR(img):
    img = load_image(img)
    pred = inference_model.predict(tf.expand_dims(img, axis=0))
    pred = decode_pred(pred)
    return pred

st.title('Urdu OCR')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    prediction = Urdu_OCR(image)
    st.write('Prediction: ', prediction)
