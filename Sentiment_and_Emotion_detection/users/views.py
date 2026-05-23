import os
import csv
import io
import uuid
import json
import re
import math
from pathlib import Path
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from .forms import UploadCSVForm, ClassifyForm
from .models import UploadedDataset
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import Gemini client (google-genai)
try:
    from google import genai
except Exception:
    genai = None

GEMINI_KEY = settings.GEMINI_API_KEY

# Utility: simple preprocessing
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
NON_PRINTABLE = re.compile(r'[^\x20-\x7E]+')
SPECIAL_CHARS = re.compile(r'[^0-9A-Za-z\s]')

def preprocess_text(s: str) -> str:
    if s is None:
        return ''
    s = s.strip()
    s = s.lower()
    s = URL_RE.sub(' <URL> ', s)
    s = EMAIL_RE.sub(' <EMAIL> ', s)
    s = NON_PRINTABLE.sub(' ', s)
    s = SPECIAL_CHARS.sub(' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# Save uploaded file & return session id + path
def save_uploaded_file(f):
    session_id = uuid.uuid4().hex
    media_dir = Path(settings.MEDIA_ROOT) / 'uploads'
    media_dir.mkdir(parents=True, exist_ok=True)
    file_path = media_dir / f"{session_id}.csv"
    with open(file_path, 'wb') as out:
        for chunk in f.chunks():
            out.write(chunk)
    return session_id, str(file_path)

# Read CSV and preprocess
def read_and_preprocess(path, text_column=None):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # choose text column
    if text_column is None:
        # heuristics: find column named 'text' or first string-like column
        candidates = [c for c in df.columns if 'text' in c.lower()]
        if candidates:
            text_column = candidates[0]
        else:
            text_column = df.columns[0]
    df['cleaned_text'] = df[text_column].astype(str).apply(preprocess_text)
    return df, text_column

# Gemini call wrapper
def call_gemini_for_text(text):
    """
    Returns dict: {'sentiment':..., 'emotion':..., 'sentiment_conf':..., 'emotion_conf':...}
    Uses google-genai client if available; requires GEMINI_API_KEY in env.
    """
    # fallback simple heuristic if no client
    if not genai or not GEMINI_KEY:
        # simple rule-based fallback (very coarse)
        low = {'sad','angry','terrible','bad','hate','awful','worst','disappointed'}
        high = {'love','great','awesome','amazing','good','wonderful','best','happy'}
        t = text.lower()
        sentiment = 'Neutral'
        emotion = 'neutral'
        s_conf = 0.6
        e_conf = 0.6
        for w in high:
            if w in t:
                sentiment = 'Positive'
                emotion = 'joy'
                s_conf = 0.85; e_conf = 0.8
                break
        for w in low:
            if w in t:
                sentiment = 'Negative'
                emotion = 'sadness'
                s_conf = 0.82; e_conf = 0.78
                break
        return {
            'sentiment': sentiment,
            'emotion': emotion,
            'sentiment_conf': s_conf,
            'emotion_conf': e_conf,
            'raw': None,
        }

    # If genai client present, call Gemini 2.5 Flash
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = (
        "You are an assistant that returns only a single JSON object. "
        "Analyze the following text and return keys: sentiment, emotion, sentiment_conf, emotion_conf.\n\n"
        f"Text: '''{text}'''\n\n"
        "Sentiment must be one of: Positive, Negative, Neutral. "
        "Emotion must be one of: joy, sadness, anger, fear, surprise, disgust, neutral.\n\n"
        "Return JSON only, for example: {\"sentiment\":\"Positive\",\"emotion\":\"joy\",\"sentiment_conf\":0.95,\"emotion_conf\":0.92}\n"
    )
    try:
        # Use the models.generate_content path (docs show similar pattern).
        response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
        # response parsing depends on client. We'll try to read text content.
        text_out = ""
        # different SDK versions may nest content differently:
        try:
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    text_out += part.text
        except Exception:
            # fallback
            try:
                text_out = response.candidates[0].content[0].text
            except Exception:
                text_out = str(response)
        # extract JSON substring
        json_str = None
        try:
            start = text_out.index('{')
            end = text_out.rindex('}') + 1
            json_str = text_out[start:end]
        except Exception:
            json_str = None
        if json_str:
            obj = json.loads(json_str)
            return {
                'sentiment': obj.get('sentiment', 'Neutral'),
                'emotion': obj.get('emotion', 'neutral'),
                'sentiment_conf': float(obj.get('sentiment_conf', obj.get('sentiment_confidence', 0.0) or 0.0)),
                'emotion_conf': float(obj.get('emotion_conf', obj.get('emotion_confidence', 0.0) or 0.0)),
                'raw': text_out,
            }
        else:
            return {'sentiment': 'Neutral', 'emotion': 'neutral', 'sentiment_conf': 0.0, 'emotion_conf': 0.0, 'raw': text_out}
    except Exception as e:
        # on error, return conservative fallback
        return {'sentiment': 'Neutral', 'emotion': 'neutral', 'sentiment_conf': 0.0, 'emotion_conf': 0.0, 'raw': str(e)}

# Home
def home(request):
    return render(request, 'home.html')

# Preprocess view
def preprocess_view(request):
    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data['file']
            session_id, path = save_uploaded_file(f)
            df, text_col = read_and_preprocess(path)
            # save cleaned csv
            cleaned_path = Path(settings.MEDIA_ROOT) / 'uploads' / f"{session_id}_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            # Save metadata
            UploadedDataset.objects.create(original_filename=f.name, stored_path=str(cleaned_path), row_count=len(df))
            request.session['last_session'] = session_id
            request.session['cleaned_path'] = str(cleaned_path)
            request.session['text_column'] = text_col
            # show first 50
            first50 = df[['cleaned_text']].head(50).to_dict(orient='records')
            return render(request, 'preview.html', {'rows': first50, 'count': len(df), 'session_id': session_id})
    else:
        form = UploadCSVForm()
    return render(request, 'preprocess.html', {'form': form})

# Preview view (if user wants to come back)
def preview_view(request):
    cleaned_path = request.session.get('cleaned_path')
    if not cleaned_path:
        return redirect('preprocess')
    df = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False)
    first50 = df[['cleaned_text']].head(50).to_dict(orient='records')
    return render(request, 'preview.html', {'rows': first50, 'count': len(df), 'session_id': request.session.get('last_session')})
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

COMMON_ASPECTS = [
    "battery", "battery life", "battery backup", "charging", "charging speed",
    "camera", "camera quality", "picture quality", "image quality",
    "screen", "display", "brightness", "resolution",
    "sound", "audio", "speaker", "sound quality",
    "performance", "speed", "lag", "smoothness",
    "design", "build quality", "body", "material",
    "price", "value", "value for money"
]


def visualize_view(request):
    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        max_rows = int(request.POST.get('max_rows', '0') or 0)

        if form.is_valid():
            f = form.cleaned_data['file']
            session_id, path = save_uploaded_file(f)

            # Load and preprocess data
            df, text_col = read_and_preprocess(path)
            if max_rows > 0:
                df = df.head(max_rows)

            results = []
            sentiments, emotions = [], []
            sent_conf, emo_conf = [], []
            cleaned_texts = []

            client = None
            if genai and GEMINI_KEY:
                client = genai.Client(api_key=GEMINI_KEY)

            # CLASSIFY EACH ROW
            for _, row in df.iterrows():
                cleaned = row['cleaned_text']
                r = call_gemini_for_text(cleaned)

                sentiments.append(r['sentiment'])
                emotions.append(r['emotion'])
                sent_conf.append(r['sentiment_conf'])
                emo_conf.append(r['emotion_conf'])
                cleaned_texts.append(cleaned)

                results.append({
                    "cleaned_text": cleaned,
                    "sentiment": r['sentiment'],
                    "emotion": r['emotion'],
                    "sentiment_conf": r['sentiment_conf'],
                    "emotion_conf": r['emotion_conf'],
                })

            # Count data
            sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
            emotion_counts = pd.Series(emotions).value_counts().to_dict()

            charts_dir = Path(settings.MEDIA_ROOT) / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            sid = uuid.uuid4().hex

            # ====================================================================
            # 1. SENTIMENT DONUT CHART
            # ====================================================================
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            labs = list(sentiment_counts.keys())
            vals = [sentiment_counts[k] for k in labs]
            ax1.pie(vals, labels=labs, wedgeprops=dict(width=0.35), autopct='%1.1f%%')
            donut_path = charts_dir / f"donut_{sid}.png"
            fig1.savefig(donut_path)
            plt.close(fig1)

            # ====================================================================
            # 2. EMOTION HORIZONTAL BAR
            # ====================================================================
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ek, ev = list(emotion_counts.keys()), [emotion_counts[k] for k in emotion_counts]
            ax2.barh(ek, ev, color="steelblue")
            emotion_h_path = charts_dir / f"emotion_h_{sid}.png"
            fig2.savefig(emotion_h_path)
            plt.close(fig2)

            # ====================================================================
            # 3. SENTIMENT KDE
            # ====================================================================
            fig3 = plt.figure(figsize=(7, 4))
            sns.kdeplot(sent_conf, fill=True, color='green')
            plt.title("Sentiment Confidence Density")
            plt.xlabel("Confidence")
            sent_kde_path = charts_dir / f"sent_kde_{sid}.png"
            fig3.savefig(sent_kde_path)
            plt.close(fig3)

            # ====================================================================
            # 4. EMOTION KDE
            # ====================================================================
            fig4 = plt.figure(figsize=(7, 4))
            sns.kdeplot(emo_conf, fill=True, color='purple')
            plt.title("Emotion Confidence Density")
            plt.xlabel("Confidence")
            emo_kde_path = charts_dir / f"emo_kde_{sid}.png"
            fig4.savefig(emo_kde_path)
            plt.close(fig4)

            # ====================================================================
            # 5. SCATTER PLOT
            # ====================================================================
            fig5, ax5 = plt.subplots(figsize=(7, 4))
            ax5.scatter(sent_conf, emo_conf, alpha=0.6)
            ax5.set_xlabel("Sentiment Confidence")
            ax5.set_ylabel("Emotion Confidence")
            ax5.set_title("Sentiment vs Emotion Confidence")
            scatter_path = charts_dir / f"scatter_{sid}.png"
            fig5.savefig(scatter_path)
            plt.close(fig5)

            # ====================================================================
            # 6. WORD CLOUD POSITIVE
            # ====================================================================
            pos_text = " ".join([cleaned_texts[i] for i in range(len(results)) if sentiments[i] == "Positive"]) or "none"
            fig_wc1 = WordCloud(width=800, height=400, colormap="Greens").generate(pos_text)
            pos_wc_path = charts_dir / f"pos_wc_{sid}.png"
            plt.imshow(fig_wc1); plt.axis("off")
            plt.savefig(pos_wc_path, bbox_inches='tight'); plt.close()

            # ====================================================================
            # 7. WORD CLOUD NEGATIVE
            # ====================================================================
            neg_text = " ".join([cleaned_texts[i] for i in range(len(results)) if sentiments[i] == "Negative"]) or "none"
            fig_wc2 = WordCloud(width=800, height=400, colormap="Reds").generate(neg_text)
            neg_wc_path = charts_dir / f"neg_wc_{sid}.png"
            plt.imshow(fig_wc2); plt.axis("off")
            plt.savefig(neg_wc_path, bbox_inches='tight'); plt.close()

            # ====================================================================
            # 8. ASPECT-BASED SENTIMENT
            # ====================================================================
            aspect_map = {}

            def add_aspect(a, s):
                a = a.lower().strip()
                if a not in aspect_map:
                    aspect_map[a] = {"Positive": 0, "Neutral": 0, "Negative": 0}
                if s in aspect_map[a]:
                    aspect_map[a][s] += 1

            # Gemini Extraction
            if client:
                for clean in cleaned_texts:
                    aspect_prompt = f"""
Extract aspects ONLY as JSON.

Text: "{clean}"

Return format:
{{
 "camera": "Positive",
 "battery life": "Negative"
}}
"""
                    try:
                        asp_res = client.models.generate_content(
                            model="gemini-2.5-flash", contents=[aspect_prompt]
                        )
                        tx = asp_res.text.strip()
                        match = re.search(r"\{.*\}", tx, re.DOTALL)
                        asp_json = json.loads(match.group()) if match else {}
                        for asp, sent in asp_json.items():
                            add_aspect(asp, sent)
                    except:
                        pass

            # Rule fallback
            for clean in cleaned_texts:
                t = clean.lower()
                for asp in COMMON_ASPECTS:
                    if asp in t:
                        sent = "Neutral"
                        if any(w in t for w in ["good", "love", "great", "excellent"]):
                            sent = "Positive"
                        elif any(w in t for w in ["bad", "poor", "worst", "terrible"]):
                            sent = "Negative"
                        add_aspect(asp, sent)

            # Plot
            figA, axA = plt.subplots(figsize=(8, 5))
            aspect_path = charts_dir / f"aspect_{sid}.png"

            if len(aspect_map) == 0:
                plt.text(0.3, 0.5, "Not Enough Aspect Data", fontsize=20)
                plt.savefig(aspect_path); plt.close()
            else:
                aspects = list(aspect_map.keys())
                pos = [aspect_map[a]["Positive"] for a in aspects]
                neu = [aspect_map[a]["Neutral"] for a in aspects]
                neg = [aspect_map[a]["Negative"] for a in aspects]

                axA.barh(aspects, pos, color="#4e79a7", label="Positive")
                axA.barh(aspects, neu, left=pos, color="#bab0ac", label="Neutral")
                axA.barh(aspects, neg, left=[pos[i]+neu[i] for i in range(len(pos))], color="#e15759", label="Negative")

                axA.set_title("Aspect-Based Sentiment")
                figA.savefig(aspect_path, bbox_inches="tight")
                plt.close(figA)


            # ====================================================================
            # Save CSV
            # ====================================================================
            annotated_df = pd.DataFrame(results)
            annotated_path = charts_dir / f"annotated_{sid}.csv"
            annotated_df.to_csv(annotated_path, index=False)

            # ====================================================================
            # Store in Session
            # ====================================================================
            request.session['last_viz'] = {
                "sid": sid,
                "donut": f"/media/charts/{donut_path.name}",
                "emotion_h": f"/media/charts/{emotion_h_path.name}",
                "sent_kde": f"/media/charts/{sent_kde_path.name}",
                "emo_kde": f"/media/charts/{emo_kde_path.name}",
                "scatter": f"/media/charts/{scatter_path.name}",
                "pos_wc": f"/media/charts/{pos_wc_path.name}",
                "neg_wc": f"/media/charts/{neg_wc_path.name}",
                "aspect": f"/media/charts/{aspect_path.name}",
                "annotated_csv": f"/media/charts/{annotated_path.name}",
            }

            return redirect("visualize_result", session_id=sid)

    # GET request
    form = UploadCSVForm()
    return render(request, "visualize.html", {"form": form})

def visualize_result(request, session_id):
    last = request.session.get('last_viz')
    if not last or last.get('sid') != session_id:
        return HttpResponse("No visualization found for this session. Please run visualize.")
    return render(request, 'visualize_result.html', {'viz': last})

# Classify single text
# def classify_view(request):
#     result = None
#     if request.method == 'POST':
#         form = ClassifyForm(request.POST)
#         if form.is_valid():
#             text = form.cleaned_data['text']
#             cleaned = preprocess_text(text)
#             res = call_gemini_for_text(cleaned)
#             result = {
#                 'original': text,
#                 'cleaned': cleaned,
#                 'sentiment': res['sentiment'],
#                 'emotion': res['emotion'],
#                 'sentiment_conf': res['sentiment_conf'],
#                 'emotion_conf': res['emotion_conf'],
#                 'raw': res.get('raw'),
#             }
#     else:
#         form = ClassifyForm()
#     return render(request, 'classify.html', {'form': form, 'result': result})


# Classify single text (UPDATED WITH SHAP-LIKE EXPLANATION)
def classify_view(request):
    result = None
    if request.method == 'POST':
        form = ClassifyForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            cleaned = preprocess_text(text)

            # Main prediction
            res = call_gemini_for_text(cleaned)

            result = {
                'original': text,
                'cleaned': cleaned,
                'sentiment': res['sentiment'],
                'emotion': res['emotion'],
                'sentiment_conf': res['sentiment_conf'],
                'emotion_conf': res['emotion_conf'],
                'raw': res.get('raw'),
            }

            # ===================================================================
            # 🔥 PSEUDO-SHAP EXPLANATION (LLM-friendly version)
            # ===================================================================

            words = cleaned.split()
            shap_scores = {}

            # Compute importance by removing each word
            for w in words:
                modified_text = " ".join([x for x in words if x != w])
                r2 = call_gemini_for_text(modified_text)

                # Score: change in sentiment → high impact
                if r2['sentiment'] != res['sentiment']:
                    score = 1.0
                else:
                    score = 0.25

                shap_scores[w] = score

            # Sort top 10 important words
            top_words = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)[:10]

            # ===================================================================
            # 🔥 Draw SHAP-like bar plot
            # ===================================================================
            charts_dir = Path(settings.MEDIA_ROOT) / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            shap_path = charts_dir / f"shap_{uuid.uuid4().hex}.png"

            # Plot
            labels = [w[0] for w in top_words]
            values = [w[1] for w in top_words]

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.barh(labels, values, color="#4e79a7")
            ax.invert_yaxis()
            ax.set_title("Word Importance (Pseudo-SHAP)")
            ax.set_xlabel("Impact Score")

            plt.tight_layout()
            plt.savefig(shap_path)
            plt.close(fig)

            # Add SHAP plot to result
            result['shap_graph'] = f"/media/charts/{shap_path.name}"

    else:
        form = ClassifyForm()

    return render(request, 'classify.html', {'form': form, 'result': result})

