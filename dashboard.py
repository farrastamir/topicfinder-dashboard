import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import os
import io
from collections import Counter
import re

st.set_page_config(layout="wide")
st.title("📰 Topic Summary Dashboard (ZIP Berisi CSV)")

@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            st.error("❌ Tidak ada file .csv dalam ZIP.")
            return []
        dfs = []
        for f in csv_files:
            with zip_ref.open(f) as file:
                try:
                    df = pd.read_csv(file, delimiter=';', quotechar='"', on_bad_lines='skip', engine='python')
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Gagal membaca {f}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if 'show_wordcloud' not in st.session_state:
    st.session_state['show_wordcloud'] = False
if 'dynamic_wordcloud' not in st.session_state:
    st.session_state['dynamic_wordcloud'] = True

st.markdown("### 📁 Pilih sumber data ZIP")
input_type = st.radio("Input ZIP via:", ["Upload File", "Link Download"])

zip_data = None
if input_type == "Upload File":
    uploaded = st.file_uploader("Unggah file ZIP", type="zip")
    if uploaded:
        zip_data = uploaded
else:
    zip_url = st.text_input("Masukkan URL file ZIP")
    if st.button("Download ZIP"):
        if zip_url:
            try:
                tmp_path = "/tmp/downloaded.zip"
                urllib.request.urlretrieve(zip_url, tmp_path)
                zip_data = tmp_path
            except Exception as e:
                st.error(f"❌ Gagal mengunduh: {e}")

if 'last_df' not in st.session_state:
    st.session_state['last_df'] = None

if zip_data:
    with st.spinner("Membaca dan memproses data..."):
        df = extract_csv_from_zip(zip_data)
        if not df.empty:
            st.session_state['last_df'] = df.copy()

if st.session_state['last_df'] is not None:
    df = st.session_state['last_df']
    for col in ['title', 'body', 'url', 'sentiment']:
        df[col] = df[col].astype(str).str.strip("'")

    df['label'] = df['label'].fillna('')
    df['tier'] = df['tier'].fillna('-')
    df['tier'] = pd.Categorical(df['tier'], categories=['Tier 1', 'Tier 2', 'Tier 3', '-', ''], ordered=True)
    all_labels = sorted(set([label.strip() for sub in df['label'] for label in sub.split(',') if label.strip()]))
    sentiments_all = sorted(df['sentiment'].str.lower().unique())

    # Sidebar (statistik + filter)
    with st.sidebar:
        st.markdown("### 📊 Statistik")
        sentiments = df['sentiment'].str.lower()
        st.markdown(f"**📰 Total Artikel:** {df.shape[0]}")
        st.markdown(f"<span style='color:green;'>🟢 Positif:</span> {sum(sentiments == 'positive')}", unsafe_allow_html=True)
        st.markdown(f"<span style='color:gray;'>⚪ Netral:</span> {sum(sentiments == 'neutral')}", unsafe_allow_html=True)
        st.markdown(f"<span style='color:red;'>🔴 Negatif:</span> {sum(sentiments == 'negative')}", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🔍 Filter")
        sentiment_filter = st.selectbox("Sentimen", options=["All"] + sentiments_all)
        keyword_input = st.text_input("Kata kunci (\"frasa\" -exclude)")
        label_filter = st.selectbox("Label", options=["All"] + all_labels)
        st.session_state['show_wordcloud'] = st.checkbox("Tampilkan Word Cloud", value=st.session_state['show_wordcloud'])
        if st.session_state['show_wordcloud']:
            st.session_state['dynamic_wordcloud'] = st.checkbox("Word Cloud Dinamis", value=st.session_state['dynamic_wordcloud'])
        highlight_words = st.text_input("Highlight Kata")

    filtered_df = df.copy()
    if sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'].str.lower() == sentiment_filter]
    if label_filter != 'All':
        filtered_df = filtered_df[filtered_df['label'].apply(lambda x: label_filter in [s.strip() for s in x.split(',')])]

    def parse_advanced_keywords(query):
        query = query.strip()
        if not query:
            return [], [], []
        include_groups, exclude_words, exact_phrases = [], [], []
        token_pattern = r'\"[^\"]+\"|\([^\)]+\)|\S+'
        tokens = re.findall(token_pattern, query)
        for tok in tokens:
            if tok.startswith('"') and tok.endswith('"'):
                exact_phrases.append(tok.strip('"'))
            elif tok.startswith('-'):
                inner = tok[1:].strip()
                exclude_words.extend(inner.strip('()').split())
            elif tok.startswith('(') and tok.endswith(')'):
                or_group = [w.strip() for w in tok.strip('()').split('OR') if w.strip()]
                include_groups.append(or_group)
            else:
                include_groups.append([tok.strip()])
        return include_groups, exact_phrases, exclude_words

    def match_advanced(text, includes, phrases, excludes):
        text = text.lower()
        if any(word in text for word in excludes):
            return False
        for phrase in phrases:
            if phrase.lower() not in text:
                return False
        for group in includes:
            if not any(word.lower() in text for word in group):
                return False
        return True

    if keyword_input:
        includes, phrases, excludes = parse_advanced_keywords(keyword_input)
        mask = filtered_df['title'].apply(lambda x: match_advanced(x, includes, phrases, excludes)) | \
               filtered_df['body'].apply(lambda x: match_advanced(x, includes, phrases, excludes))
        filtered_df = filtered_df[mask]

    highlight_tokens = re.findall(r'\"[^\"]+\"|\S+', highlight_words)
    highlight_words_set = set([h.strip('"').lower() for h in highlight_tokens])

    def highlight_text(text):
        words = text.split()
        highlighted = [f"<mark>{w}</mark>" if any(hw in w.lower() for hw in highlight_words_set) else w for w in words]
        return ' '.join(highlighted)

    # Custom function to get URL based on tier preference
    def get_best_link(sub_df):
        for tier in ['Tier 1', 'Tier 2', 'Tier 3', '-', '']:
            result = sub_df[sub_df['tier'] == tier]['url']
            if not result.empty:
                return result.iloc[0]
        return '-'

    grouped = filtered_df.groupby('title').agg(
        Article=('title', 'count'),
        Sentiment=('sentiment', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
        Link=('title', lambda x: get_best_link(filtered_df[filtered_df['title'] == x.iloc[0]]))
    ).reset_index().sort_values(by='Article', ascending=False)

    def sentiment_color(sent):
        s = sent.lower()
        if s == 'positive': return f'<span style="color:green;font-weight:bold">{s}</span>'
        if s == 'negative': return f'<span style="color:red;font-weight:bold">{s}</span>'
        if s == 'neutral': return f'<span style="color:gray;font-weight:bold">{s}</span>'
        return sent

    grouped['Sentiment'] = grouped['Sentiment'].apply(sentiment_color)
    grouped['Link'] = grouped['Link'].apply(lambda x: f'<a href="{x}" target="_blank">Link</a>' if x != '-' else '-')

    st.markdown("### 📊 Ringkasan Topik")
    st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
    st.write(grouped[['title', 'Article', 'Sentiment', 'Link']].to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state['show_wordcloud']:
        st.markdown("""
        <style>
        .wordcloud-container { position: fixed; top: 60px; right: 0; width: 30%; height: 100%; overflow-y: scroll; background-color: #111; padding: 1rem; color: white; }
        </style>
        <div class="wordcloud-container">
        <h4>☁️ Word Cloud (Top 500)</h4>
        """, unsafe_allow_html=True)

        base_df = filtered_df if st.session_state['dynamic_wordcloud'] else df
        all_text = ' '.join(base_df['title'].tolist() + base_df['body'].tolist())
        tokens = re.findall(r'\b\w{3,}\b', all_text.lower())
        stop_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
        common_stopwords = set(pd.read_csv(stop_url, header=None)[0].tolist())
        tokens = [word for word in tokens if word not in common_stopwords]
        word_freq = Counter(tokens).most_common(500)
        wc_df = pd.DataFrame(word_freq, columns=['Kata', 'Jumlah'])
        st.markdown(wc_df.to_html(index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Silakan upload atau unduh ZIP untuk melihat ringkasan topik.")
