import streamlit as st
from PIL import Image
import io
import json

# Import your processing functions
from invoice_ocr import process_invoice_image

st.set_page_config(page_title="Invoice OCR & JSON Extractor", layout="centered")

st.title("📄 Invoice OCR & JSON Extractor")
st.write("Upload an invoice image, and the app will OCR it and extract structured JSON.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an invoice image file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Read image bytes only once
    img_bytes = uploaded_file.read()

    # Display uploaded image
    st.subheader("Uploaded Image")
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(image, use_column_width=True)
    except Exception as e:
        st.error(f"Could not display image: {e}")
        st.stop()

    # Run processing
    st.subheader("Extraction Progress")
    with st.spinner("Processing image, please wait..."):
        # Save temporarily
        temp_path = "temp_invoice.png"
        image.save(temp_path)

        try:
            result = process_invoice_image(temp_path)
            st.success("Extraction complete!")
        except Exception as e:
            st.error(f"Error during extraction: {e}")
            st.stop()

    # Show JSON result
    st.subheader("Extracted JSON")
    st.text_area("Resulting JSON", json.dumps(result, indent=2), height=300)

    st.download_button(
        label="Download JSON",
        data=json.dumps(result, indent=2),
        file_name="invoice_extraction.json",
        mime="application/json"
    )
else:
    st.info("Please upload an invoice image to begin.")
