import streamlit as st
from PIL import Image
import torch
from torchvision import transforms as T
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
from torchvision import models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Ensure the image has three channels (for RGB images)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img_array = img_array / 255.0  # Normalize the pixel values
    img_array = np.transpose(img_array, (2, 0, 1))  # Change the order to (channels, height, width)
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


# Function to deprocess the image
def deprocess(img):
    img = img[0].permute(1, 2, 0)
    img = img * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])
    img = np.clip(img.numpy(), 0, 1)
    return img


# Function to make predictions
def predict(image):
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image)
        image = image.to(device)
        prediction = model(image)
        return prediction.cpu().numpy().squeeze()


# Load the modified model
model = models.resnet18()
model.fc = nn.Sequential(
    nn.Linear(512, 14),
    nn.Sigmoid()
)

# Load the state_dict, skipping layers with mismatched keys
state_dict_path = 'Pneumonia_model.pt'
state_dict = torch.load(state_dict_path, map_location=device)
model_dict = model.state_dict()

# Filter out unnecessary keys
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

# Update the current model's state_dict
model_dict.update(state_dict)

# Load the updated state_dict into the model
model.load_state_dict(model_dict)

model.to(device)
model.eval()

# Sample pathology_list
pathology_list = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Nodule', 'Pneumothorax', 'Atelectasis',
                  'Pleural_Thickening', 'Mass', 'Edema', 'Consolidation', 'Infiltration', 'Fibrosis', 'Pneumonia']

# Streamlit app
st.title('Lung Disease Prediction App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=300)

    # Make predictions when the button is clicked
    if st.button('Predict'):
        # st.info(f"**Pathology List**: *{pathology_list}*")
        st.divider()
        prediction = predict(image)
        class_labels = list(np.where(prediction == prediction.max())[0])
        title = itemgetter(*class_labels)(pathology_list)

        # Display the probability values
        # st.write('Ground Truth : {}'.format(prediction))
        probability_data = pd.DataFrame({'Pathology Class': pathology_list, 'Probability': prediction})
        st.write("## Prediction probabilities")
        col1, col2 = st.columns([1.6, 2.2])
        with col1:
            st.write(probability_data)
        with col2:
            st.bar_chart(probability_data.set_index('Pathology Class'))
        st.success(f'Predicted Pathology Class: ***{title}***')
else:
    st.warning("Please upload an image to proceed.") 
