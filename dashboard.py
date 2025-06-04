import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import os
import io
from collections import Counter
import re
import itertools

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

def parse_advanced_keywords(query):
    query = query.strip()
    if not query:
        return [], [], []

    include_groups = []
    exclude_words = []
    exact_phrases = []

    if ' OR ' in query:
        parts = query.split(' OR ')
        or_combinations = []
        for part in parts:
            groups, phrases, excludes = parse_advanced_keywords(part)
            or_combinations.append((groups, phrases, excludes))
        return 'OR_COMBO', or_combinations, []

    token_pattern = r'"[^"]+"|\([^)]+\)|\S+'
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
    if includes == 'OR_COMBO':
        for group_set, phrase_set, exclude_set in phrases:
            if match_advanced(text, group_set, phrase_set, exclude_set):
                return True
        return False

    if any(word in text for word in excludes):
        return False
    for phrase in phrases:
        if phrase.lower() not in text:
            return False
    for group in includes:
        if not any(word.lower() in text for word in group):
            return False
    return True

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
    df['tier'] = df['tier'].fillna('-')
    df['tier'] = pd.Categorical(df['tier'], categories=['Tier 1', 'Tier 2', 'Tier 3', '-', ''], ordered=True)
    all_labels = sorted(set([label.strip() for sub in df['label'] for label in sub.split(',') if label.strip()]))
    sentiments_all = sorted(df['sentiment'].str.lower().unique())

    col_stat, col_filter, col_word = st.columns([1, 1.3, 1.7])

    with col_filter:
        st.markdown("#### 🔍 Filter")
        sentiment_filter = st.selectbox("Sentimen", options=["All"] + sentiments_all)
        keyword_input = st.text_input("Kata kunci (\"frasa\" -exclude)")
        label_filter = st.selectbox("Label", options=["All"] + all_labels)
        dynamic_wordcloud = st.checkbox("Word Cloud Dinamis", value=True)

    filtered_df = df.copy()
    if sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'].str.lower() == sentiment_filter]
    if label_filter != 'All':
        filtered_df = filtered_df[filtered_df['label'].apply(lambda x: label_filter in [s.strip() for s in x.split(',')])]
    if keyword_input:
        includes, phrases, excludes = parse_advanced_keywords(keyword_input)
        mask = filtered_df['title'].apply(lambda x: match_advanced(x, includes, phrases, excludes)) | \
               filtered_df['body'].apply(lambda x: match_advanced(x, includes, phrases, excludes))
        filtered_df = filtered_df[mask]

    sentiments = filtered_df['sentiment'].str.lower()
    total_artikel = filtered_df.shape[0]
    total_positif = sum(sentiments == 'positive')
    total_negatif = sum(sentiments == 'negative')
    total_netral = sum(sentiments == 'neutral')

    with col_stat:
        st.markdown("""
        ### 🧾 **Total Artikel:**
        <span style='font-size:26px; font-weight:bold;'>{}</span><br>
        <span style='color:green;'>🟢 {} Positif</span> &nbsp;&nbsp;
        <span style='color:gray;'>⚪ {} Netral</span> &nbsp;&nbsp;
        <span style='color:red;'>🔴 {} Negatif</span>
        """.format(total_artikel, total_positif, total_netral, total_negatif), unsafe_allow_html=True)

    grouped = filtered_df.groupby('title').agg(
        Article=('title', 'count'),
        Sentiment=('sentiment', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
        Link=('url', lambda x: x[df.loc[x.index].sort_values(by='tier').index[0]] if not x.dropna().empty else '-')
    ).reset_index().sort_values(by='Article', ascending=False)

    def color_sentiment(s):
        if s.lower() == 'positive': return f'<span style="color:green;font-weight:bold">{s}</span>'
        if s.lower() == 'negative': return f'<span style="color:red;font-weight:bold">{s}</span>'
        return f'<span style="color:gray;font-weight:bold">{s}</span>'

    grouped['Sentiment'] = grouped['Sentiment'].apply(color_sentiment)
    grouped['Link'] = grouped['Link'].apply(lambda x: f'<a href="{x}" target="_blank">Lihat</a>' if x != '-' else '-')
    grouped['title'] = grouped['title'].apply(lambda x: f'<div title="{x}" style="max-width:400px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{x}</div>')

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

    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        st.write(grouped.to_html(escape=False, index=False), unsafe_allow_html=True)

    with col2:
        st.markdown("### ☁️ Word Cloud (Top 500 Kata)")
        base_df = filtered_df if dynamic_wordcloud else df
        all_text = ' '.join(base_df['title'].tolist() + base_df['body'].tolist())
        tokens = re.findall(r'\b\w{3,}\b', all_text.lower())
        common_stopwords = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt", header=None)[0].tolist())
        tokens = [word for word in tokens if word not in common_stopwords]
        word_freq = Counter(tokens).most_common(500)
        wc_df = pd.DataFrame(word_freq, columns=['Kata', 'Jumlah'])
        st.dataframe(wc_df, use_container_width=True)
else:
    st.info("Silakan upload atau unduh ZIP untuk melihat ringkasan topik.")
