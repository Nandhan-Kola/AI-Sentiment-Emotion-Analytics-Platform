🧠 AI-Powered Sentiment & Emotion Analytics Platform
A Django-based web application that detects sentiment polarity and emotions from text using Google Gemini LLM — no training data required, with interactive visualizations and explainable AI.
________________________________________
🔍 Problem Statement
Traditional sentiment analysis tools rely on large labeled datasets, domain-specific training, and complex model pipelines — making them expensive, inflexible, and hard to interpret. This platform eliminates those limitations by leveraging Google Gemini (LLM) for zero-shot sentiment and emotion classification, making it scalable, domain-independent, and accessible to non-technical users.
________________________________________
✨ Key Features
•	📂 CSV Dataset Upload — Upload any text dataset and analyze thousands of records automatically
•	🧹 Automated Text Preprocessing — Removes URLs, emails, special characters, and noise
•	💬 Sentiment Classification — Classifies text as Positive, Negative, or Neutral
•	❤️ Emotion Detection — Detects emotions: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
•	📊 Rich Visualizations — Donut charts, bar graphs, KDE plots, word clouds, scatter plots
•	🔍 Aspect-Based Sentiment Analysis — Identifies sentiment for specific aspects (camera, battery, price, etc.)
•	⚡ Real-Time Single Text Classification — Instantly analyze any input text with confidence scores
•	🧾 Explainable AI (Pseudo-SHAP) — Highlights which words influenced the prediction most
•	📥 Download Annotated Dataset — Export results as a labeled CSV file
________________________________________
🖥️ Application Screenshots
Home Page	Preprocessing	Visualization
Upload CSV, Preprocess, Classify	Cleaned text preview with 2000+ rows	Sentiment donut + emotion bar charts
Aspect-based sentiment chart, word clouds, and SHAP-style word importance graphs also included.
________________________________________
🛠️ Tech Stack
Category	Technology
Backend Framework	Django 4.2
AI / LLM	Google Gemini 2.5 Flash (via google-genai API)
Data Processing	Pandas, NumPy, Regex
Visualization	Matplotlib, Seaborn, WordCloud
Explainability	Pseudo-SHAP (word-level importance)
Database	SQLite (Django default)
Frontend	HTML, CSS (Django Templates)
Environment	Python 3.8+
________________________________________
📁 Project Structure
Sentiment_and_Emotion_detection/
│
├── manage.py                          # Django entry point
├── db.sqlite3                         # SQLite database
├── req.txt                            # Python dependencies
│
├── Sentiment_and_Emotion_detection/   # Main Django app settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── detection/                         # Core application
    ├── views.py                       # All logic: preprocessing, classification, visualization
    ├── models.py                      # UploadedDataset model
    ├── forms.py                       # Upload & classify forms
    ├── urls.py
    └── templates/
        ├── home.html
        ├── preprocess.html
        ├── preview.html
        ├── visualize.html
        ├── visualize_result.html
        └── classify.html
________________________________________
🚀 How to Run Locally
1. Clone the repository
git clone https://github.com/Nandhan-Kola/sentiment-emotion-detection.git
cd sentiment-emotion-detection
2. Install dependencies
pip install -r req.txt
3. Set up your Gemini API key
Create a .env file in the root folder:
GEMINI_API_KEY=your_google_gemini_api_key_here
Get your free API key at: Google AI Studio
4. Run database migrations
python manage.py migrate
5. Start the server
python manage.py runserver
6. Open in browser
http://127.0.0.1:8000
________________________________________
🔄 How It Works
1.	User uploads a CSV file containing text data
2.	System preprocesses the text — removes noise, normalizes, cleans
3.	Gemini LLM classifies each row for sentiment and emotion (zero-shot)
4.	Visualizations generated — charts, word clouds, aspect-based analysis
5.	Single text mode — real-time classification with SHAP-style word importance
6.	Results downloadable as an annotated CSV
________________________________________
📊 Visualizations Generated
•	🍩 Sentiment distribution donut chart
•	📊 Emotion frequency horizontal bar chart
•	📈 Sentiment confidence KDE (density) plot
•	📉 Emotion confidence KDE plot
•	🔵 Sentiment vs Emotion confidence scatter plot
•	☁️ Word cloud for Positive texts
•	☁️ Word cloud for Negative texts
•	📌 Aspect-based sentiment stacked bar chart
•	🧮 Pseudo-SHAP word importance bar chart
________________________________________
✅ Testing Summary
Test Case	Scenario	Status
TC-01	Upload valid CSV	✅ Pass
TC-04	Detect Positive sentiment	✅ Pass
TC-05	Detect Negative sentiment	✅ Pass
TC-07	Detect Joy emotion	✅ Pass
TC-09	Batch analysis (100+ rows)	✅ Pass
TC-14	Aspect-based sentiment	✅ Pass
________________________________________

📄 References
•	Pang et al. (2002) — Thumbs Up? Sentiment Classification Using ML Techniques
•	Bing Liu (2012) — Sentiment Analysis and Opinion Mining
•	Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers
•	Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)
•	Google DeepMind (2023) — Gemini: A Family of Highly Capable Multimodal Models
________________________________________
⭐ If you found this project useful, consider giving it a star!

