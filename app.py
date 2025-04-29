# --- START OF FILE app.py ---

import streamlit as st
import requests
from PIL import Image
import io
import json
import time
import threading
# No need for atexit with daemon threads
from prometheus_client import start_http_server, Counter, Histogram, Gauge, REGISTRY
import os

# --- Initialize Metrics ONCE using Session State ---
# Use a unique key in session state to track initialization
if 'prometheus_metrics_initialized' not in st.session_state:
    print("Initializing Prometheus metrics...") # For debugging

    # Define metrics and store them in session state
    st.session_state.PROMETHEUS_UPLOADS_COUNTER = Counter(
        'streamlit_3d_defect_uploads_total',
        'Total number of images uploaded',
        registry=REGISTRY # Explicitly use the default registry
    )
    st.session_state.PROMETHEUS_ANALYSIS_REQUESTS_COUNTER = Counter(
        'streamlit_3d_defect_analysis_requests_total',
        'Total number of analysis requests initiated',
        registry=REGISTRY
    )
    st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER = Counter(
        'streamlit_3d_defect_analysis_results_total',
        'Total number of analysis results processed, labeled by status',
        ['status'],
        registry=REGISTRY
    )
    st.session_state.PROMETHEUS_PREDICTIONS_COUNTER = Counter(
        'streamlit_3d_defect_predictions_total',
        'Total predictions made, labeled by result type',
        ['result'],
        registry=REGISTRY
    )
    st.session_state.PROMETHEUS_ANALYSIS_API_LATENCY_SECONDS = Histogram(
        'streamlit_3d_defect_analysis_api_latency_seconds',
        'Latency of the API call to the defect detection backend',
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf")],
        registry=REGISTRY
    )
    # Add other metrics here if you have them (like the Gauge example)

    # Mark as initialized
    st.session_state.prometheus_metrics_initialized = True
    print("Prometheus metrics initialized and stored in session state.")

# --- Prometheus Server Setup ---
PROMETHEUS_PORT = 8001
# Use a simple global flag for the thread start, as session_state might not be ideal
# for managing threads across potential multiple worker processes in the future.
_prometheus_server_started_flag = "_prometheus_server_started" # Use a string key

# Check if the flag exists globally (handling potential multi-threading access cautiously)
if not globals().get(_prometheus_server_started_flag, False):
    # Set the flag immediately to prevent race conditions if possible
    globals()[_prometheus_server_started_flag] = True

    def start_metrics_server_thread():
        """Target function to run the Prometheus server."""
        # Check environment variable within the thread too if needed
        if os.getenv("ENABLE_PROMETHEUS_METRICS", "true").lower() == "true":
            try:
                print(f"Attempting to start Prometheus metrics endpoint on port {PROMETHEUS_PORT}...")
                start_http_server(port=PROMETHEUS_PORT, registry=REGISTRY)
                print(f"Prometheus metrics endpoint running on http://localhost:{PROMETHEUS_PORT}/metrics")
            except OSError as e:
                # Handle cases where the port might already be in use by another process/instance
                print(f"Warning: Could not start Prometheus metrics endpoint on port {PROMETHEUS_PORT}: {e}")
                # Optionally show a warning in the UI if Streamlit context is available,
                # but might be tricky from a separate thread without passing `st` object.
                # Consider logging this properly.
            except Exception as e:
                print(f"Error starting Prometheus metrics endpoint thread: {e}")
        else:
            print("Prometheus metrics export disabled via environment variable.")

    # Start the server thread only once
    _prometheus_thread = threading.Thread(
        target=start_metrics_server_thread,
        daemon=True # Essential: thread exits when main process exits
    )
    _prometheus_thread.start()
    print("Prometheus metrics server thread initiated.")


# --- Page Configuration ---
# Must be the first Streamlit command after imports and setup
st.set_page_config(
    page_title="3D Print Inspector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:help@example.com',
        'Report a bug': "mailto:bugs@example.com",
        'About': """
        ## 3D Print Inspector

        This app uses AI to detect defects in 3D printed objects.

        Developed with Streamlit.
        """
    }
)

# --- Constants ---
API_URL = "http://localhost:5001/predict"
EXAMPLE_IMAGE_PATH = None

# --- Helper Functions ---
# MODIFY process_image and display_result to use metrics from session_state

def process_image(image_bytes):
    """Sends image to API, returns result, and records metrics."""
    files = {"image": image_bytes}
    start_time = time.time()
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        latency = time.time() - start_time
        # Access metric from session_state
        st.session_state.PROMETHEUS_ANALYSIS_API_LATENCY_SECONDS.observe(latency)

        response.raise_for_status()
        # Access metric from session_state
        st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER.labels(status='success').inc()
        return response.json()

    except requests.exceptions.ConnectionError as e:
        st.error(f"❌ Connection Error: Could not connect to the analysis server at {API_URL}.")
        # Access metric from session_state
        st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER.labels(status='connection_error').inc()
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Timeout Error: The analysis server took too long to respond.")
        # Access metric from session_state
        st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER.labels(status='timeout').inc()
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request Error: An error occurred: {e}")
        status_label = 'http_error' if hasattr(e, 'response') and e.response is not None else 'request_error'
        # Access metric from session_state
        st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER.labels(status=status_label).inc()
        return None
    except json.JSONDecodeError:
        st.error("❌ Response Error: The analysis server returned an invalid response.")
        # Access metric from session_state
        st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER.labels(status='json_decode_error').inc()
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during analysis: {e}")
        # Access metric from session_state
        st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER.labels(status='unknown_error').inc()
        return None


def display_result(result_data, result_placeholder):
    """Displays the analysis result and records prediction metrics."""
    with result_placeholder.container():
        st.subheader("🔬 Analysis Result")
        prediction = result_data.get('prediction', None)
        confidence = result_data.get('confidence', None)

        if prediction == 1: # Non-Defective
            st.markdown("### <span style='color: #28a745;'>✅ STATUS: NON-DEFECTIVE</span>", unsafe_allow_html=True)
            st.success("The 3D print appears to be properly formed without significant visual defects.")
            # Access metric from session_state
            st.session_state.PROMETHEUS_PREDICTIONS_COUNTER.labels(result='non_defective').inc()
            if confidence is not None:
                 st.metric(label="Confidence", value=f"{100*confidence:.2f}%")
            st.balloons()
        elif prediction == 0: # Defective
            st.markdown("### <span style='color: #dc3545;'>❌ STATUS: DEFECTIVE</span>", unsafe_allow_html=True)
            st.error("Potential defects detected in the 3D print. Further inspection is recommended.")
            # Access metric from session_state
            st.session_state.PROMETHEUS_PREDICTIONS_COUNTER.labels(result='defective').inc()
            if confidence is not None:
                st.metric(label="Confidence", value=f"{100*confidence:.2f}%")
        else:
            st.warning("⚠️ Unexpected Result: The analysis returned an unknown status.")
            # Access metric from session_state
            st.session_state.PROMETHEUS_PREDICTIONS_COUNTER.labels(result='unknown').inc()
            st.json(result_data)

        with st.expander("Show Raw Analysis Data"):
            st.json(result_data)

# --- Sidebar ---
# (Sidebar code remains the same)
with st.sidebar:
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=200) # Example logo
    st.title("ℹ️ About & Help")
    st.markdown("---")
    st.info("""
        **Welcome to the 3D Print Inspector!**

        This tool helps you quickly assess the quality of your 3D prints.

        **How it works:**
        1.  Upload an image of your print.
        2.  Click 'Analyze Print'.
        3.  Review the AI-powered analysis.
    """)
    st.markdown("---")
    with st.expander("📖 User Manual", expanded=False):
        st.markdown(f"""
        **1. Uploading Your Image:**
           - Click the 'Browse files' button or drag and drop an image file (`.jpg`, `.jpeg`, `.png`).
           - **Tip:** Use clear, well-lit images focusing on the potentially problematic areas of the print for best results. Avoid blurry or distant shots.

        **2. Starting the Analysis:**
           - Once the image preview appears, click the 'Analyze Print 🚀' button.
           - Please wait while the image is processed. This may take a few seconds depending on the model and server load.

        **3. Understanding the Results:**
           - **✅ NON-DEFECTIVE:** The AI did not detect common visual defects. The print is likely okay based on the analysis.
           - **❌ DEFECTIVE:** The AI detected patterns associated with common 3D printing defects (like warping, stringing, under-extrusion, etc.). Manual inspection is recommended.
           - **Confidence:** This score indicates how confident the AI is in its prediction (0-100%). Higher is generally better.
           - **Raw Data:** Expand this section to see the complete output from the analysis server (useful for debugging).

        **Troubleshooting:**
           - **Connection Error:** Ensure the backend analysis server is running and accessible at the configured URL (`{API_URL}`).
           - **Timeout Error:** The server is taking too long. Try again later or check the server status.
           - **Other Errors:** If you encounter persistent issues, please use the 'Report a bug' option in the menu (⋮).
           - **Metrics Server:** If metrics aren't appearing in Prometheus, check the Streamlit console output for errors related to starting the metrics server on port {PROMETHEUS_PORT}.
        """)
    st.markdown("---")
    st.caption("Version 1.0.2 (Metrics Fix)")


# --- Main Application Area ---
st.title("🖨️ 3D Printing Defect Detection")
st.markdown("Upload a clear image of your 3D printed object to check for potential defects using AI.")
st.markdown("---")

# --- Step 1: Upload ---
st.subheader("1. Upload Image")
uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, and PNG formats. Use a clear, well-lit photo of the print."
)

result_placeholder = st.empty()

if uploaded_file is not None:
    # --- Increment upload counter ---
    # Access metric from session_state
    st.session_state.PROMETHEUS_UPLOADS_COUNTER.inc()

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("🖼️ Image Preview")
        img_bytes = None # Initialize
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

            img_bytes_io = io.BytesIO()
            save_format = image.format if image.format in ['JPEG', 'PNG'] else 'PNG'
            image.save(img_bytes_io, format=save_format)
            img_bytes = img_bytes_io.getvalue()

        except Exception as e:
            st.error(f"Error loading image: {e}")
            # Access metric from session_state
            st.session_state.PROMETHEUS_ANALYSIS_RESULTS_COUNTER.labels(status='image_error').inc()

    with col2:
        st.subheader("2. Analyze Print")
        st.markdown("Click the button below to start the defect analysis.")

        if img_bytes:
            analyze_button = st.button("Analyze Print 🚀", key="analyze", type="primary", use_container_width=True)

            if analyze_button:
                # --- Increment analysis request counter ---
                # Access metric from session_state
                st.session_state.PROMETHEUS_ANALYSIS_REQUESTS_COUNTER.inc()

                with st.spinner("🔍 Analyzing image... Please wait."):
                    api_result = process_image(img_bytes)

                if api_result:
                    display_result(api_result, result_placeholder)
        else:
            st.warning("⚠️ Please upload a valid image file first.")

else:
    result_placeholder.info("☝️ Upload an image using the browser above to get started.")

# --- Footer ---
st.markdown("---")
st.caption("Disclaimer: This AI analysis provides a preliminary assessment. Always perform manual inspection for critical prints.")

# --- END OF FILE app.py ---