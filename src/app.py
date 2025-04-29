import streamlit as st
import requests
from PIL import Image
import io
import json

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="3D Printing Defect Detection",
    page_icon="🖨️",
    layout="wide"
)

st.title("3D Printing Defect Detection")
st.write("Upload an image to check for printing defects")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image of a 3D printed object...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded 3D Print", use_column_width=True)
    
    # When the user clicks the predict button
    if st.button("Analyze for Defects"):
        with st.spinner("Analyzing image..."):
            # Convert the file to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            
            # API endpoint URL (adjust to your actual API endpoint)
            api_url = "http://localhost:5001/predict"
            
            # Send POST request to the API
            files = {"image": img_bytes}
            try:
                response = requests.post(api_url, files=files)
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Display the prediction with color
                    result = response.json()
                    prediction = result.get('prediction', None)
                    
                    with col2:
                        st.subheader("Analysis Result")
                        
                        if prediction == 1:
                            st.markdown("### :green[NON-DEFECTIVE] ✅")
                            st.markdown("The 3D print appears to be properly formed without defects.")
                        elif prediction == 0:
                            st.markdown("### :red[DEFECTIVE] ❌")
                            st.markdown("The 3D print shows signs of defects that need attention.")
                        else:
                            st.warning("Unexpected prediction value")
                            
                        # Display additional details if available
                        if 'confidence' in result:
                            st.metric("Confidence Score", f"{100-result['confidence']:.2f}%")
                        
                        st.json(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")

# Add some additional information about the model
with st.expander("About this Defect Detection System"):
    st.write("""
    This application uses a machine learning model trained to detect defects in 3D printed objects.
    
    - Upload a clear image of your 3D print
    - Click "Analyze for Defects" to get the result
    - Green "NON-DEFECTIVE" result means the print is good
    - Red "DEFECTIVE" result means issues were detected
    """)
