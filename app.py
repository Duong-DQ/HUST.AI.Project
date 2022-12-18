import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

agemodel = tf.keras.models.load_model("Age-Gender-Prediction\checkpoints\agemodel.h5")
genmodel = tf.keras.models.load_model("Age-Gender-Prediction\checkpoints\genmodel.h5")

def process_and_predict(file):
    im = Image.open(file)
    width, height = im.size
    if width == height:
        im = im.resize((200, 200), Image.ANTIALIAS)
    else:
        if width > height:
            left = width / 2 - height / 2
            right = width / 2 + height / 2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.ANTIALIAS)

    ar = np.asarray(im)
    ar = ar.astype("float32")
    ar /= 255.0
    ar = ar.reshape(-1, 200, 200, 3)

    age = agemodel.predict(ar)
    gender = np.round(genmodel.predict(ar))
    if gender == 0:
        gender = "male"
    elif gender == 1:
        gender = "female"

    st.markdown("Age:", int(age), "\n Gender:", gender)
    return im.resize((300, 300), Image.ANTIALIAS)


def main():
    st.header("AGE AND GENDER PREDICTION")
    st.write(
        "Upload an image of yourself below to find out! (preferably a squared image and containing only your face)"
    )
    file = st.file_uploader("Upload Photo")
    if file is not None:
        st.image(file, width=300)
        process_and_predict(file)
        

if __name__ == "__main__":
    main()
