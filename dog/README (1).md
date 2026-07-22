# 🐕 Dog Skin Disease Classifier

A professional Streamlit web app that uses a custom-trained CNN model to classify dog skin diseases from an uploaded image. The model is loaded directly from Hugging Face Hub, so no large files need to live in this GitHub repo.

**Model:** [Adnan-official/DOG_SKIN_DISEASE](https://huggingface.co/Adnan-official/DOG_SKIN_DISEASE)

## ✨ Features

- Clean, modern UI with custom styling
- Upload a dog skin image (JPG/PNG) and get instant predictions
- Confidence score + full class-probability bar chart
- Auto-downloads model weights from Hugging Face Hub (cached after first load)
- Deployable for free on Streamlit Community Cloud

## 📁 Project Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🚀 Run Locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. Push this folder to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo, branch, and set the main file to `app.py`.
4. Click **Deploy**. First load will take a minute or two while the model downloads from Hugging Face.

No API keys or secrets are required since the Hugging Face repo is public.

## ⚠️ Disclaimer

This app is for educational/demo purposes only and is **not** a substitute for professional veterinary diagnosis. Always consult a licensed veterinarian for accurate diagnosis and treatment.

## 🛠️ Tech Stack

- **TensorFlow / Keras** — CNN model
- **Streamlit** — web app framework
- **Hugging Face Hub** — model hosting & download
- **Pillow / NumPy** — image preprocessing
