import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    layout="wide"
)

st.title("🍎 Fruits & Vegetables Image Classification")
st.write("Upload one or more images. Similar fruits and vegetables will be grouped automatically.")

# -------------------- LOAD MODEL --------------------
MODEL_PATH = r"C:\Users\Sangeeta\OneDrive\Documents\Projects\Image_classifications\Image_classify.keras"

model = load_model(MODEL_PATH)

# -------------------- CLASS NAMES --------------------
data_cat = [
    'apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon'
]

# -------------------- IMAGE SIZE --------------------
img_height = 180
img_width = 180

# -------------------- FILE UPLOADER --------------------
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------- PREDICTION --------------------
if uploaded_files:

    grouped_images = {}

    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):

        # Load image
        image = tf.keras.utils.load_img(
            file,
            target_size=(img_height, img_width)
        )

        # Convert image to array
        img_array = tf.keras.utils.img_to_array(image)

        # Add batch dimension
        img_batch = tf.expand_dims(img_array, 0)

        # Predict
        prediction = model.predict(img_batch, verbose=0)

        # Convert logits to probabilities
        score = tf.nn.softmax(prediction)

        predicted_class = data_cat[np.argmax(score)]

        confidence = float(np.max(score) * 100)

        # Store grouped images
        if predicted_class not in grouped_images:
            grouped_images[predicted_class] = []

        grouped_images[predicted_class].append({
            "image": image,
            "filename": file.name,
            "confidence": confidence
        })

        # Update progress bar
        progress_bar.progress((i + 1) / len(uploaded_files))

    progress_bar.empty()

    st.success(f"Successfully classified {len(uploaded_files)} image(s).")

    st.markdown("---")

    # -------------------- DISPLAY GROUPS --------------------
    for category in sorted(grouped_images.keys()):

        images = grouped_images[category]

        st.subheader(f"📂 {category.title()} ({len(images)})")

        cols = st.columns(4)

        for idx, item in enumerate(images):

            with cols[idx % 4]:

                st.image(
                    item["image"],
                    use_container_width=True
                )

                st.caption(item["filename"])

                st.write(f"Confidence: {item['confidence']:.2f}%")

        st.markdown("---")

else:
    st.info("Please upload one or more fruit/vegetable images.")