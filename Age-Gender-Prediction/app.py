import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

agemodel = tf.keras.models.load_model(".\\checkpoints\\agemodel-3.h5")
genmodel = tf.keras.models.load_model(".\\checkpoints\\genmodel-3.h5")
genmodel1 = tf.keras.models.load_model(".\\checkpoints\\genmodel-1.h5")
genmodel2 = tf.keras.models.load_model(".\\checkpoints\\genmodel-2.h5")
agemodel1 = tf.keras.models.load_model(".\\checkpoints\\agemodel-1.h5")
agemodel2 = tf.keras.models.load_model(".\\checkpoints\\agemodel-2.h5")

def process_and_predict(file):
    im = Image.open(file)
    im = im.convert("RGB")
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

    age = int((agemodel.predict(ar) + agemodel.predict(ar) + agemodel.predict(ar))/3)
    gender = np.round(genmodel.predict(ar))
    if gender == 0:
        gender = "Male"
    elif gender == 1:
        gender = "Female"

    info = "Age: " + str(int(age)) + " - Gender: " + gender
    st.subheader(info)

    # age = agemodel1.predict(ar)
    # gender = np.round(genmodel1.predict(ar))
    # if gender == 0:
    #     gender = "Male"
    # elif gender == 1:
    #     gender = "Female"

    # info = "Age: " + str(int(age)) + " - Gender: " + gender
    # st.subheader(info)

    # age = agemodel2.predict(ar)
    # gender = np.round(genmodel2.predict(ar))
    # if gender == 0:
    #     gender = "Male"
    # elif gender == 1:
    #     gender = "Female"

    # info = "Age: " + str(int(age)) + " - Gender: " + gender
    # st.subheader(info)

    return im.resize((300, 300), Image.ANTIALIAS)


def main():
    
    st.header("AGE AND GENDER PREDICTION")

    with st.expander("This is AI-course project of group 3"):
        st.markdown('''
        Our project is built to predict the age and gender of humans using CNN.

        Upload an image find out! (preferably a squared image and containing only a face)

        [link to repo]()
        ''')
    file = st.file_uploader("Upload Photo")
    if file is not None:
        st.image(file, width=300)
        process_and_predict(file)
        

if __name__ == "__main__":
    main()
