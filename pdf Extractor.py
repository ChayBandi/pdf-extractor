#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers pymupdf Pillow torch torchvision torchaudio')


# In[3]:


pip install pdf2image transformers pillow


# In[4]:


pip install keras==2.14.0  # Last stable version before Keras 3


# In[7]:


pip install streamlit


# In[9]:


import tensorflow as tf
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))


# In[11]:


with tf.device('/GPU:0'):
    import streamlit as st
    import pymupdf  # PyMuPDF for PDF processing
import os
from transformers import pipeline
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Initialize AI models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF using PyMuPDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    doc = pymupdf.open(pdf_path)
    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text("text") + "\n"
    return extracted_text.strip()

# Function to extract and save images from the PDF
def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    Extracts images from a PDF and saves them as PNG files.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder to save extracted images.

    Returns:
        list: List of saved image file paths.
    """
    doc = pymupdf.open(pdf_path)
    image_paths = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):
        page = doc[page_num]
        img_list = page.get_images(full=True)

        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_name = f"{output_folder}/page_{page_num+1}_img_{img_index+1}.{img_ext}"

            # Save image to file
            with open(img_name, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(img_name)

    return image_paths

# Function to summarize extracted text
def generate_summary(text):
    """
    Generates a summary of extracted text using the BART-large model.

    Args:
        text (str): The input text to summarize.

    Returns:
        str: The summarized text.
    """
    chunk_size = 1024
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = []
    for chunk in chunks:
        max_length = min(300, max(len(chunk.split()) // 2, 100))
        min_length = max(max_length // 3, 50)

        summary_result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary_result[0]['summary_text'])

    return " ".join(summaries)

# Function for Question Answering
def answer_question(context, question):
    """
    Uses a Hugging Face QA model to answer questions from the extracted text.

    Args:
        context (str): The text from which to answer the question.
        question (str): The user's question.

    Returns:
        str: The answer.
    """
    response = qa_pipeline(question=question, context=context)
    return response["answer"]

# Function to generate an explanation for images
def explain_images(image_paths):
    """
    Uses an AI model to generate an explanation for each image.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        dict: Dictionary containing image explanations.
    """
    explanations = {}
    for img_path in image_paths:
        img = Image.open(img_path)
        explanation = image_captioning_pipeline(img)[0]["generated_text"]
        explanations[img_path] = explanation
    return explanations
'''
# Function to display images and explanations in console
def display_images(image_paths, explanations):
    """
    Displays extracted images using matplotlib and prints explanations.

    Args:
        image_paths (list): List of image file paths.
        explanations (dict): Dictionary of image explanations.
    """
    for img_path in image_paths:
        img = Image.open(img_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Extracted Image: {img_path}")
        plt.figtext(0.5, 0.01, explanations.get(img_path, "No explanation available"), wrap=True, horizontalalignment='center', fontsize=10)
        plt.show()
        print(f"\nImage: {img_path}\nExplanation: {explanations.get(img_path, 'No explanation available')}")
        print("-" * 50)

# Function to save summary and image explanations to a text file
def save_results_to_file(summary, explanations, output_file="summary_and_explanations.txt"):
    """
    Saves the generated summary and image explanations to a text file.

    Args:
        summary (str): The summarized text.
        explanations (dict): Dictionary containing image explanations.
        output_file (str): The output file name.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("DETAILED SUMMARY\n")
        file.write("-" * 50 + "\n")
        file.write(summary + "\n\n")

        file.write("IMAGE EXPLANATIONS\n")
        file.write("-" * 50 + "\n")
        for img_path, explanation in explanations.items():
            file.write(f"Image: {img_path}\nExplanation: {explanation}\n")
            file.write("-" * 50 + "\n")

    print(f"\nResults saved to {output_file}")

# Main function to handle the workflow
def main():
    pdf_path = input("Enter the path to the PDF file: ")

    try:
        # Extract text from PDF
        print("\nExtracting text from the PDF...")
        extracted_text = extract_text_from_pdf(pdf_path)

        # Summarize the text
        if extracted_text:
            print("\nGenerating detailed summary...")
            summary = generate_summary(extracted_text)
            print("\nDetailed Summary:")
            print("-" * 50)
            print(summary)
        else:
            print("\nNo text found in the PDF!")

        # Extract images from PDF
        print("\nExtracting images from the PDF...")
        image_paths = extract_images_from_pdf(pdf_path)
        if image_paths:
            print(f"\nExtracted {len(image_paths)} images. Generating explanations...")
            explanations = explain_images(image_paths)
            display_images(image_paths, explanations)
        else:
            print("\nNo images found in the PDF!")

        # Save results to a file
        save_results_to_file(summary, explanations)

        # Ask user for a question related to the text
        question = input("\nEnter a question related to the document: ")
        if question.strip():
            answer = answer_question(extracted_text, question)
            print("\nAnswer to your question:")
            print("-" * 50)
            print(answer)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
'''
# Streamlit UI
st.title("📄 AI-Powered PDF Analyzer")
st.write("Upload a PDF to extract text, summarize, answer questions, and analyze images.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file
    pdf_path = "uploaded_pdf.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text
    with st.spinner("Extracting text..."):
        extracted_text = extract_text_from_pdf(pdf_path)
    
    if extracted_text:
        st.subheader("Extracted Text")
        st.text_area("PDF Text", extracted_text, height=300)

        # Generate Summary
        with st.spinner("Generating Summary..."):
            summary = generate_summary(extracted_text)
        st.subheader("Summary")
        st.write(summary)

        # Question Answering
        st.subheader("Ask a Question about the Document")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Finding the answer..."):
                    answer = qa_pipeline(question=question, context=extracted_text)
                st.success(f"**Answer:** {answer['answer']}")
            else:
                st.warning("Please enter a question.")

    else:
        st.warning("No text found in the PDF!")

    # Extract Images
    with st.spinner("Extracting images..."):
        image_paths = extract_images_from_pdf(pdf_path)
    
    if image_paths:
        st.subheader("Extracted Images and AI Descriptions")
        explanations = explain_images(image_paths)

        for img_path in image_paths:
            img = Image.open(img_path)
            st.image(img, caption=f"Extracted Image: {img_path}", use_column_width=True)
            st.write(f"**Description:** {explanations.get(img_path, 'No description available')}")

else:
    st.info("Please upload a PDF to begin.")


# In[16]:


#streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py


# In[ ]:




