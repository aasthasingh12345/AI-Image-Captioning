import streamlit as st
import tensorflow as tf
from PIL import Image
import Caption_it

# Define the Streamlit app
def app():
    # Set the title and the page icon
    st.set_page_config(page_title='Image Captioning', page_icon=':camera:')

    # Set the app header
    st.title('Image Captioning')
    st.write('Upload an image and the app will generate a caption for it.')

    # Upload the image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg'])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=400, width=300)
        img=uploaded_file.name
        # Generate the caption
        
        caption = Caption_it.get_caption(img)
        st.write('**Caption:**', caption)

# Run the app
if __name__ == '__main__':
    app()
