# 🧠 AI-Powered Image Denoising using Autoencoder

## 📌 Project Overview
This project is a deep learning-based image denoising system using an autoencoder. It learns to remove noise from images and reconstruct clean outputs from noisy inputs.

---

## 🎯 Objective
To build a neural network that:
- Takes noisy images as input
- Learns meaningful compressed features (latent space)
- Reconstructs clean images as output

---

## 🧠 Model Architecture
- Encoder: Compresses image into latent representation  
- Decoder: Reconstructs image from latent space  

---

## 🔄 Workflow
Noisy Image → Encoder → Latent Space → Decoder → Denoised Image

---

## 📊 Dataset
- MNIST dataset (handwritten digits)
- Grayscale images (28×28)
- Noise added during training

---

## ⚙️ Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib
- Google Colab

---

## 🚀 Run the Project

### 🔗 Google Colab Notebook (Recommended)
Due to large model size and training requirements, the complete project is implemented and trained in Google Colab.

👉 Click here to run the project:
https://colab.research.google.com/drive/1PSP34FKn0PzeZELY6Pfj891lwXVY-_Ab?usp=sharing
## 🧪 Training Details
- Loss Function: MSE + L1 Loss
- Optimizer: Adam
- Epochs: 30–100 (best results around ~55 epochs)
- Input: Noisy images
- Output: Clean images

---

## 📉 Observations
- Best performance observed at ~55 epochs
- Higher epochs may lead to overfitting and blur
- Balanced loss improves sharpness

---

## 💾 Model File
Trained model file (`.pth`) is large and not uploaded to GitHub due to size limits.  
It is stored in Google Drive.

---

## 🔗 Model Download
(Add your Google Drive link here if needed)

---

## 🚀 Future Improvements
- Use GANs or Diffusion Models for sharper results
- Add SSIM loss for better image quality
- Deploy using Streamlit web app
- Train on real-world noisy images

---

## 🧠 Key Learnings
- Autoencoder architecture
- Latent space representation
- Image denoising techniques
- Overfitting and optimization
- End-to-end deep learning workflow

---

## 👩‍💻 Author
- Name: Sudha C Parimala
- Project Type: Academic 

---

## 📌 Conclusion
This project demonstrates how autoencoders can be used for image restoration by learning compressed representations and reconstructing clean images from noisy inputs.
