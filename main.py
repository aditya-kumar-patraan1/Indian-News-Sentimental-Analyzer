import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
from collections import Counter
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian News Sentiment Anaylzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background: #0f0f13; }
.block-container { padding: 2rem 3rem; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-label { font-size: 0.78rem; color: #7a7a9d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 2rem; font-weight: 700; color: #e8e8ff; }
.metric-sub   { font-size: 0.82rem; color: #9a9ab0; margin-top: 4px; }

/* Sentiment badges */
.badge-pos { background:#0d3d2b; color:#4ade80; border:1px solid #166534; border-radius:99px; padding:3px 14px; font-size:.82rem; font-weight:600; }
.badge-neu { background:#1e2a3a; color:#93c5fd; border:1px solid #1e40af; border-radius:99px; padding:3px 14px; font-size:.82rem; font-weight:600; }
.badge-neg { background:#3d0d0d; color:#f87171; border:1px solid #991b1b; border-radius:99px; padding:3px 14px; font-size:.82rem; font-weight:600; }

/* Predict box */
.predict-box {
    background: linear-gradient(135deg, #1a1a2e, #0f0f1f);
    border: 1px solid #3a3a6a;
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.result-positive { border-left: 5px solid #4ade80; }
.result-neutral  { border-left: 5px solid #93c5fd; }
.result-negative { border-left: 5px solid #f87171; }

/* Section headings */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #e8e8ff;
    margin: 2rem 0 1rem;
    border-bottom: 1px solid #2a2a4a;
    padding-bottom: .5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0a12;
    border-right: 1px solid #1e1e3a;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
STOPWORDS = {
    'the','a','an','in','of','to','and','is','for','on','at','by','with',
    'as','be','are','was','this','that','it','its','from','or','but','not',
    'has','have','had','will','he','she','they','their','our','we','you',
    'his','her','been','into','more','than','also','after','over','new',
    'up','out','about','who','all','one','what','your'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compound_to_label(c):
    if c > 0.05:   return 'Positive'
    if c < -0.05:  return 'Negative'
    return 'Neutral'

def get_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'F1 Score':  round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }

COLORS = {'Positive': '#4ade80', 'Neutral': '#93c5fd', 'Negative': '#f87171'}
BADGE  = {'Positive': 'badge-pos', 'Neutral': 'badge-neu', 'Negative': 'badge-neg'}

plt.rcParams.update({
    'figure.facecolor': '#0f0f13',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#2a2a4a',
    'text.color':       '#e8e8ff',
    'axes.labelcolor':  '#c0c0e0',
    'xtick.color':      '#9090b0',
    'ytick.color':      '#9090b0',
    'grid.color':       '#2a2a4a',
    'axes.titlecolor':  '#e8e8ff',
})

# ── Session state ─────────────────────────────────────────────────────────────
for key in ['trained', 'lr', 'rf', 'tfidf', 'df', 'y_test', 'y_pred_lr', 'y_pred_rf']:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.setdefault('trained', False)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Indian News Sentiment Anaylzer")
    st.caption("News Headline Sentiment Analysis")
    st.divider()

    st.markdown("### 📁 Upload Dataset")
    uploaded = st.file_uploader("Excel or CSV file", type=['xlsx', 'csv'])

    st.divider()
    st.markdown("### ⚙️ Embedding Settings")
    max_features = st.slider("Max Sequence Length", 100, 2000, 500, 100)
    ngram_max    = st.radio("Convolutional Kernels", ["Size 3", "Size 3 & 5"], index=1)
    ngram        = (1, 1) if "Size 3" in ngram_max else (1, 2)

    st.divider()
    st.markdown("### 🧬 Network Parameters")
    n_estimators = st.slider("Training Epochs", 10, 100, 50, 10)

    st.divider()
    st.markdown("### 🔀 Train / Test Split")
    test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100

    train_btn = st.button("🚀 Train Models", use_container_width=True, type="primary")

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Indian News Sentiment Anaylzer")
st.markdown("Upload your dataset, train both models, explore visualisations, and predict live headlines.")

# ── Load & train ──────────────────────────────────────────────────────────────
if train_btn:
    if uploaded is None:
        st.error("Please upload a dataset first.")
    else:
        with st.spinner("Loading & training architectures…"):
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                # Drop junk columns
                drop_cols = [c for c in df.columns if c.lower() in ['index','unnamed: 0','s_no']]
                df.drop(columns=drop_cols, inplace=True)

                # Find headline column
                headline_col = None
                for c in df.columns:
                    if 'headline' in c.lower():
                        headline_col = c
                        break
                if headline_col is None:
                    st.error("Could not find a 'Headline' column. Make sure your file has one.")
                    st.stop()

                df.dropna(subset=[headline_col], inplace=True)
                df.reset_index(drop=True, inplace=True)
                df['Headline_clean'] = df[headline_col].apply(clean_text)

                # Sentiment label
                if 'Compound' in df.columns:
                    df['Sentiment'] = df['Compound'].apply(compound_to_label)
                elif 'Sentiment' in df.columns:
                    pass
                else:
                    st.error("Need either a 'Compound' or 'Sentiment' column to create labels.")
                    st.stop()

                # Split & vectorise
                X_train, X_test, y_train, y_test = train_test_split(
                    df['Headline_clean'], df['Sentiment'],
                    test_size=test_size, random_state=42, stratify=df['Sentiment']
                )
                tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram, stop_words='english')
                X_tr  = tfidf.fit_transform(X_train)
                X_te  = tfidf.transform(X_test)

                # Train
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_tr, y_train)

                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                rf.fit(X_tr, y_train)

                # Store
                st.session_state.update({
                    'trained': True, 'lr': lr, 'rf': rf, 'tfidf': tfidf, 'df': df,
                    'y_test': y_test,
                    'y_pred_lr': lr.predict(X_te),
                    'y_pred_rf': rf.predict(X_te),
                    'headline_col': headline_col,
                })
                st.success(f"✅ Models trained on {len(df)} records!")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Main content (only shown after training) ──────────────────────────────────
if not st.session_state['trained']:
    st.info("👈 Upload your dataset and click **Train Models** to get started.")
    st.stop()

df       = st.session_state['df']
lr       = st.session_state['lr']
rf       = st.session_state['rf']
tfidf    = st.session_state['tfidf']
y_test   = st.session_state['y_test']
y_pred_lr = st.session_state['y_pred_lr']
y_pred_rf = st.session_state['y_pred_rf']

tabs = st.tabs(["📊 Overview", "📈 Visualisations", "⚔️ Model Comparison", "🔮 Live Predict", "📋 Data"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    counts = df['Sentiment'].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Positive</div>
            <div class="metric-value" style="color:#4ade80">{counts.get('Positive',0)}</div>
            <div class="metric-sub">{counts.get('Positive',0)/len(df)*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Neutral</div>
            <div class="metric-value" style="color:#93c5fd">{counts.get('Neutral',0)}</div>
            <div class="metric-sub">{counts.get('Neutral',0)/len(df)*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Negative</div>
            <div class="metric-value" style="color:#f87171">{counts.get('Negative',0)}</div>
            <div class="metric-sub">{counts.get('Negative',0)/len(df)*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Performance Summary</div>', unsafe_allow_html=True)
    m_lr = get_metrics(y_test, y_pred_lr)
    m_rf = get_metrics(y_test, y_pred_rf)

    col_a, col_b = st.columns(2)
    for col, name, m, clr in [(col_a, "GoogleNet", m_lr, "#818cf8"),
                               (col_b, "ResNet",       m_rf, "#34d399")]:
        with col:
            st.markdown(f"#### {name}")
            r1, r2, r3, r4 = st.columns(4)
            for rc, label, val in zip([r1,r2,r3,r4], m.keys(), m.values()):
                rc.metric(label, f"{val:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Visualisations
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Row 1: distribution bar + pie
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        cats = ['Negative','Neutral','Positive']
        vals = [counts.get(c,0) for c in cats]
        bars = ax.bar(cats, vals, color=[COLORS[c] for c in cats], edgecolor='#2a2a4a', linewidth=0.8, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5, str(v),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e8e8ff')
        ax.set_title('Sentiment Distribution', fontsize=13, pad=12)
        ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        fig, ax = plt.subplots()
        ax.pie(vals, labels=cats, autopct='%1.1f%%',
               colors=[COLORS[c] for c in cats],
               startangle=90, wedgeprops=dict(edgecolor='#0f0f13', linewidth=2),
               textprops={'color':'#e8e8ff'})
        ax.set_title('Sentiment Share', fontsize=13, pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Row 2: compound histogram + boxplot
    if 'Compound' in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.hist(df['Compound'], bins=25, color='#818cf8', edgecolor='#0f0f13', linewidth=0.5)
            ax.axvline(0.05,  color='#4ade80', linestyle='--', linewidth=1.5, label='Positive threshold')
            ax.axvline(-0.05, color='#f87171', linestyle='--', linewidth=1.5, label='Negative threshold')
            ax.set_title('Compound Score Distribution', fontsize=13, pad=12)
            ax.set_xlabel('Compound Score')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            fig, ax = plt.subplots()
            order = ['Negative','Neutral','Positive']
            data_by_cls = [df[df['Sentiment']==s]['Compound'].values for s in order]
            bp = ax.boxplot(data_by_cls, labels=order, patch_artist=True,
                            medianprops=dict(color='white', linewidth=2),
                            whiskerprops=dict(color='#7a7a9d'),
                            capprops=dict(color='#7a7a9d'),
                            flierprops=dict(marker='o', markerfacecolor='#7a7a9d', markersize=4))
            for patch, cls in zip(bp['boxes'], order):
                patch.set_facecolor(COLORS[cls])
                patch.set_alpha(0.7)
            ax.set_title('Compound Score by Sentiment', fontsize=13, pad=12)
            ax.set_ylabel('Compound Score')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # Row 3: word count + top words
    df['word_count'] = df['Headline_clean'].apply(lambda x: len(x.split()))
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots()
        for cls in ['Negative','Neutral','Positive']:
            subset = df[df['Sentiment']==cls]['word_count']
            ax.hist(subset, bins=15, alpha=0.65, color=COLORS[cls], edgecolor='#0f0f13', label=cls)
        ax.set_title('Headline Sequence Length by Sentiment', fontsize=13, pad=12)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        all_words = ' '.join(df['Headline_clean']).split()
        filtered  = [w for w in all_words if w not in STOPWORDS and len(w) > 2]
        top_words = Counter(filtered).most_common(15)
        words, freqs = zip(*top_words)
        fig, ax = plt.subplots()
        bars = ax.barh(list(words)[::-1], list(freqs)[::-1],
                       color='#818cf8', edgecolor='#0f0f13', linewidth=0.5)
        ax.set_title('Top 15 Tokens in Dataset', fontsize=13, pad=12)
        ax.set_xlabel('Frequency')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Row 4: top TF-IDF features per class (LR coefficients)
    st.markdown('<div class="section-title">Top Feature Activations (GoogleNet)</div>', unsafe_allow_html=True)
    feature_names = tfidf.get_feature_names_out()
    classes = lr.classes_
    color_map = {'Negative':'#f87171','Neutral':'#93c5fd','Positive':'#4ade80'}
    fig, axes = plt.subplots(1, len(classes), figsize=(5*len(classes), 5))
    if len(classes) == 1: axes = [axes]
    for ax, cls in zip(axes, classes):
        coef    = lr.coef_[list(classes).index(cls)]
        top_idx = coef.argsort()[-12:][::-1]
        ax.barh(feature_names[top_idx][::-1], coef[top_idx][::-1],
                color=color_map.get(cls,'#818cf8'), edgecolor='#0f0f13', linewidth=0.5)
        ax.set_title(f'→ {cls}', fontsize=12, pad=10)
        ax.set_xlabel('Activation Weight')
    plt.suptitle('Highest Activation Weights per Sentiment Class', fontsize=13, y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Architecture Comparison</div>', unsafe_allow_html=True)

    m_lr = get_metrics(y_test, y_pred_lr)
    m_rf = get_metrics(y_test, y_pred_rf)
    comp = pd.DataFrame({'GoogleNet': m_lr, 'ResNet': m_rf})
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=1, color='#1a3a2a'), use_container_width=True)

    # Bar comparison
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        x     = np.arange(len(m_lr))
        width = 0.35
        bars1 = ax.bar(x - width/2, list(m_lr.values()), width, label='GoogleNet', color='#818cf8', edgecolor='#0f0f13')
        bars2 = ax.bar(x + width/2, list(m_rf.values()), width, label='ResNet',       color='#34d399', edgecolor='#0f0f13')
        ax.set_xticks(x)
        ax.set_xticklabels(list(m_lr.keys()), rotation=15)
        ax.set_ylim(0, 1.15)
        ax.set_title('Metric Comparison', fontsize=13, pad=12)
        ax.set_ylabel('Score')
        ax.legend()
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#e8e8ff')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Radar chart
    with c2:
        categories = list(m_lr.keys())
        N = len(categories)
        angles = [n / N * 2 * np.pi for n in range(N)] + [0]
        vals_lr = list(m_lr.values()) + [list(m_lr.values())[0]]
        vals_rf = list(m_rf.values()) + [list(m_rf.values())[0]]

        fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f13')
        ax.plot(angles, vals_lr, 'o-', linewidth=2, color='#818cf8', label='GoogleNet')
        ax.fill(angles, vals_lr, alpha=0.2, color='#818cf8')
        ax.plot(angles, vals_rf, 'o-', linewidth=2, color='#34d399', label='ResNet')
        ax.fill(angles, vals_rf, alpha=0.2, color='#34d399')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, color='#e8e8ff')
        ax.set_ylim(0, 1)
        ax.tick_params(colors='#7a7a9d')
        ax.set_title('Radar — Architecture Comparison', fontsize=12, pad=20, color='#e8e8ff')
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Confusion matrices side by side
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    label_order = ['Negative','Neutral','Positive']

    for col, name, y_pred, cmap in [
        (c1, "GoogleNet", y_pred_lr, 'Blues'),
        (c2, "ResNet",       y_pred_rf, 'Greens')
    ]:
        with col:
            cm  = confusion_matrix(y_test, y_pred, labels=label_order)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                        xticklabels=label_order, yticklabels=label_order,
                        ax=ax, linewidths=0.5, linecolor='#0f0f13',
                        cbar_kws={'shrink': 0.8})
            ax.set_title(f'Confusion Matrix — {name}', fontsize=11, pad=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # Per-class report
    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    model_choice = st.radio("Show report for:", ["GoogleNet", "ResNet"], horizontal=True)
    y_pred_show  = y_pred_lr if model_choice == "GoogleNet" else y_pred_rf
    report_dict  = classification_report(y_test, y_pred_show, output_dict=True)
    report_df    = pd.DataFrame(report_dict).T.round(4)
    st.dataframe(report_df.style.format("{:.4f}").highlight_max(axis=0, color='#1a3a2a'), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Live Predict
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Live Headline Predictor</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        user_input = st.text_area(
            "Enter one or more headlines (one per line):",
            height=160,
            placeholder="Stock markets crash as recession fears grow\nScientists celebrate major breakthrough\nGovernment releases budget report"
        )
        model_sel  = st.radio("Architecture to use:", ["GoogleNet", "ResNet", "Both"], horizontal=True)
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    with col_right:
        st.markdown("**Examples to try:**")
        st.markdown("""
- *"Scientists celebrate breakthrough in cancer research"*
- *"Markets crash amid recession fears"*
- *"Government releases annual report"*
- *"Earthquake destroys hundreds of homes"*
- *"New park opens in the heart of the city"*
        """)

    if predict_btn and user_input.strip():
        headlines = [h.strip() for h in user_input.strip().split('\n') if h.strip()]
        cleaned   = [clean_text(h) for h in headlines]
        vec       = tfidf.transform(cleaned)

        for headline, clean in zip(headlines, cleaned):
            pred_lr = lr.predict(tfidf.transform([clean]))[0]
            pred_rf = rf.predict(tfidf.transform([clean]))[0]
            prob_lr = lr.predict_proba(tfidf.transform([clean]))[0]
            prob_rf = rf.predict_proba(tfidf.transform([clean]))[0]

            if model_sel == "Both":
                display_pred = pred_lr   # primary
            elif model_sel == "GoogleNet":
                display_pred = pred_lr
            else:
                display_pred = pred_rf

            result_class = f"result-{display_pred.lower()}"
            badge_class  = BADGE[display_pred]

            st.markdown(f"""
            <div class="predict-box {result_class}">
                <div style="font-size:.85rem;color:#7a7a9d;margin-bottom:6px;">Headline</div>
                <div style="font-size:1.05rem;color:#e8e8ff;margin-bottom:12px;font-style:italic;">"{headline}"</div>
            """, unsafe_allow_html=True)

            if model_sel == "Both":
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**GoogleNet →** <span class='{BADGE[pred_lr]}'>{pred_lr}</span>", unsafe_allow_html=True)
                    for cls, p in zip(lr.classes_, prob_lr):
                        st.progress(float(p), text=f"{cls}: {p:.2%}")
                with c2:
                    st.markdown(f"**ResNet →** <span class='{BADGE[pred_rf]}'>{pred_rf}</span>", unsafe_allow_html=True)
                    for cls, p in zip(rf.classes_, prob_rf):
                        st.progress(float(p), text=f"{cls}: {p:.2%}")
            else:
                probs = prob_lr if model_sel == "GoogleNet" else prob_rf
                classes = lr.classes_ if model_sel == "GoogleNet" else rf.classes_
                st.markdown(f"**Prediction →** <span class='{badge_class}'>{display_pred}</span>", unsafe_allow_html=True)
                for cls, p in zip(classes, probs):
                    st.progress(float(p), text=f"{cls}: {p:.2%}")

            st.markdown("</div>", unsafe_allow_html=True)

    elif predict_btn:
        st.warning("Please enter at least one headline.")

    # Bulk CSV predict
    st.divider()
    st.markdown("### 📂 Bulk Predict from CSV")
    bulk_file = st.file_uploader("Upload CSV with a 'Headline' column", type=['csv'], key='bulk')
    if bulk_file:
        bulk_df  = pd.read_csv(bulk_file)
        hcol     = next((c for c in bulk_df.columns if 'headline' in c.lower()), None)
        if hcol is None:
            st.error("No 'Headline' column found in the CSV.")
        else:
            cleaned_bulk  = bulk_df[hcol].apply(clean_text)
            vec_bulk      = tfidf.transform(cleaned_bulk)
            bulk_df['GoogleNet_Prediction'] = lr.predict(vec_bulk)
            bulk_df['ResNet_Prediction'] = rf.predict(vec_bulk)
            st.dataframe(bulk_df[[hcol, 'GoogleNet_Prediction', 'ResNet_Prediction']], use_container_width=True)
            csv_out = bulk_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv_out, "predictions.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Data
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

    show_cols = [c for c in ['Headline', 'Sentiment', 'Compound', 'Positive', 'Negative', 'Neutral', 'Headline_clean'] if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=400)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Shape:**", )
        st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
    with c2:
        st.markdown("**Missing values:**")
        st.write(df[show_cols].isnull().sum().to_dict())

    csv_data = df[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Cleaned Dataset", csv_data, "cleaned_data.csv", "text/csv")