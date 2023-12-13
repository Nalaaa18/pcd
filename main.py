import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_resource()
def load_model():
	model = tf.keras.models.load_model('./kematangan_apel.hdf5',compile=False)
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Prediksi Tingkat Kematangan Buah Apel')

file = st.file_uploader("Upload gambar apel", type=["jpg", "png", "jpeg"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Tunggu sebentar....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['100%', '20%', '40%', '60%', '80%']

	class_desc = {
		'100%': 'Buah ini sudah matang saat ini',
		'20%' : 'Buah ini matang sekitar 25 hari lagi',
		'40%' : 'Buah ini matang sekitar 20 hari lagi',
		'60%' : 'Buah ini matang sekitar 15 hari lagi',
		'80%' : 'Buah ini matang sekitar 10 hari lagi'
	}

	result = class_names[np.argmax(pred)]

	output = 'Tingkat kematangan buah apel ini adalah: ' + result

	desc = 'Penjelasan: ' + class_desc[result]

	slot.text('Done')

	st.success(output)

img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)

    img_array = np.array(img)

    st.image(img)

    pred = predict_class(np.asarray(img), model)
    class_names = ['100%', '20%', '40%', '60%', '80%']
    result = class_names[np.argmax(pred)]
    output = 'Tingkat kematangan buah apel ini adalah: ' + result
    st.success(output)