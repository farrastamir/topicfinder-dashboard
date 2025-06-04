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

# Input: Upload atau URL
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

# STATE KONTROL
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
    all_labels = sorted(set([label.strip() for sub in df['label'] for label in sub.split(',') if label.strip()]))
    sentiments_all = sorted(df['sentiment'].str.lower().unique())

    # Filter
    st.sidebar.header("🔍 Filter")
    sentiment_filter = st.sidebar.selectbox("Sentimen", options=["All"] + sentiments_all)
    keyword_input = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)")
    label_filter = st.sidebar.selectbox("Label", options=["All"] + all_labels)

    filtered_df = df.copy()

    # Filter sentimen
    if sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'].str.lower() == sentiment_filter]

    # Filter label
    if label_filter != 'All':
        filtered_df = filtered_df[filtered_df['label'].apply(lambda x: label_filter in [s.strip() for s in x.split(',')])]

    # Filter kata kunci
    def match_keywords(text):
        text = text.lower()
        include_words = []
        exclude_words = []
        phrases = []
        tokens = keyword_input.split(' ')
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('"'):
                phrase = token
                while not phrase.endswith('"') and i+1 < len(tokens):
                    i += 1
                    phrase += ' ' + tokens[i]
                phrases.append(phrase.strip('"'))
            elif token.startswith('-'):
                exclude_words.append(token[1:])
            else:
                include_words.append(token)
            i += 1
        for word in exclude_words:
            if word in text:
                return False
        for word in include_words:
            if word and word not in text:
                return False
        for phrase in phrases:
            if phrase and phrase not in text:
                return False
        return True

    if keyword_input:
        mask = filtered_df['title'].apply(match_keywords) | filtered_df['body'].apply(match_keywords)
        filtered_df = filtered_df[mask]

    grouped = filtered_df.groupby('title').agg(
        Article=('title', 'count'),
        Sentiment=('sentiment', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
        Link=('url', lambda x: x.dropna().iloc[0] if not x.dropna().empty else '-')
    ).reset_index().sort_values(by='Article', ascending=False)

    sentiments = filtered_df['sentiment'].str.lower()
    st.markdown(f"""
    **Total Artikel:** {filtered_df.shape[0]} | 
    **Positif:** {sum(sentiments == 'positive')} | 
    **Negatif:** {sum(sentiments == 'negative')} | 
    **Netral:** {sum(sentiments == 'neutral')}
    """)

    def color_sentiment(s):
        if s.lower() == 'positive': return f'<span style="color:green;font-weight:bold">{s}</span>'
        if s.lower() == 'negative': return f'<span style="color:red;font-weight:bold">{s}</span>'
        return f'<span style="color:gray;font-weight:bold">{s}</span>'

    grouped['Sentiment'] = grouped['Sentiment'].apply(color_sentiment)
    grouped['Link'] = grouped['Link'].apply(lambda x: f'<a href="{x}" target="_blank">Lihat</a>' if x != '-' else '-')
    grouped['title'] = grouped['title'].apply(lambda x: f'<div title="{x}" style="max-width:400px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{x}</div>')

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 📊 Ringkasan Topik")
        st.write("(Klik judul/link untuk melihat detail)")
        st.markdown("""
        <style>
        td div {
            max-width: 500px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
        """, unsafe_allow_html=True)
        st.write(grouped.to_html(escape=False, index=False), unsafe_allow_html=True)

    with col2:
        st.markdown("### ☁️ Word Cloud (CSV)")
        all_text = ' '.join(filtered_df['title'].tolist() + filtered_df['body'].tolist())
        tokens = re.findall(r'\b\w{3,}\b', all_text.lower())
        common_stopwords = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt", header=None)[0].tolist())
        tokens = [word for word in tokens if word not in common_stopwords]
        word_freq = Counter(tokens).most_common(100)
        wc_df = pd.DataFrame(word_freq, columns=['Kata', 'Jumlah'])
        st.dataframe(wc_df)

else:
    st.info("Silakan upload atau unduh ZIP untuk melihat ringkasan topik.")
