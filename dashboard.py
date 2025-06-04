# 📦 Install dependencies
!pip install pandas ipywidgets openpyxl > /dev/null

import pandas as pd
import zipfile
import urllib.request
import os
from google.colab import files
from IPython.display import display, clear_output, HTML, Javascript
import ipywidgets as widgets

# Fungsi ekstrak CSV dari ZIP
def extract_csv_from_zip(zip_path, extract_to="/content/"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("Tidak ada file .csv dalam file .zip.")
        return [os.path.join(extract_to, f) for f in csv_files]

# Fungsi input ZIP
upload_button = widgets.Button(description="Upload File ZIP")
link_button = widgets.Button(description="Gunakan Link ZIP")
link_input = widgets.Text(placeholder='https://...', layout=widgets.Layout(width='60%'))
status_output = widgets.Output()

# Placeholder untuk menyimpan DataFrame
df = pd.DataFrame()
summary_data = pd.DataFrame()
label_filter = widgets.Dropdown(options=[], description="Label:", layout=widgets.Layout(width='200px'))

# Fungsi baca ZIP
def handle_file(zip_file):
    with status_output:
        clear_output()
        print("📂 Memproses dan membaca file... Mohon tunggu...")
    global df, summary_data
    csv_paths = extract_csv_from_zip(zip_file)
    dfs = []
    for path in csv_paths:
        try:
            dfs.append(pd.read_csv(path, delimiter=';', quotechar='"', on_bad_lines='skip', engine='python'))
        except Exception as e:
            with status_output:
                print(f"Gagal membaca {path}: {e}")
    if not dfs:
        raise ValueError("Tidak ada file CSV valid yang terbaca.")
    df = pd.concat(dfs, ignore_index=True)
    for col in ['title', 'body', 'url', 'sentiment']:
        df[col] = df[col].astype(str).str.strip("'")

    all_labels = set()
    for val in df['label'].dropna().astype(str):
        all_labels.update([x.strip() for x in val.split(',') if x.strip()])
    label_filter.options = ['All'] + sorted(all_labels)

    display(Javascript('window.scrollTo(0, 0);'))
    apply_filters()

# Upload File ZIP
upload_widget = widgets.FileUpload(accept='.zip', multiple=False)
def on_upload_clicked(b):
    if upload_widget.value:
        with status_output:
            clear_output()
            print("📤 Mengunggah file ZIP...")
        file_info = next(iter(upload_widget.value.values()))
        with open("uploaded.zip", "wb") as f:
            f.write(file_info['content'])
        handle_file("uploaded.zip")
    else:
        with status_output:
            print("⚠️ Mohon upload file terlebih dahulu.")
upload_button.on_click(on_upload_clicked)

# Link File ZIP
def on_link_clicked(b):
    url = link_input.value.strip()
    if url:
        with status_output:
            clear_output()
            print("🌐 Mengunduh file dari URL...")
        zip_file = "downloaded.zip"
        urllib.request.urlretrieve(url, zip_file)
        handle_file(zip_file)
    else:
        with status_output:
            print("⚠️ Mohon masukkan URL ZIP.")
link_button.on_click(on_link_clicked)

# Filter dan tampilan
sentiment_filter = widgets.Dropdown(options=['All'], description='Sentiment:')
keyword_input = widgets.Text(placeholder='Ketik kata kunci ("subsidi rumah" -utang)', description='Filter:', layout=widgets.Layout(width='60%'))
output_area = widgets.Output()

# CSS untuk hover dan warna sentimen
style_html = """
<style>
.output_scroll table {
  table-layout: auto;
  width: 100%;
  font-size: 13px;
  white-space: nowrap;
}
.output_scroll th, .output_scroll td {
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 250px;
  white-space: nowrap;
}
.sentiment-positive { color: green; font-weight: bold; }
.sentiment-negative { color: red; font-weight: bold; }
.sentiment-neutral { color: gray; font-weight: bold; }
</style>
"""
display(HTML(style_html))

# Filtering dan visualisasi
def apply_filters(change=None):
    clear_output(wait=True)
    display(Javascript('window.scrollTo(0, 0);'))
    display(HTML(style_html))
    display(input_layout)
    with output_area:
        clear_output()
        if df.empty:
            print("Belum ada data yang dimuat.")
            return

        filtered_df = df.copy()
        sentiment_filter.options = ['All'] + sorted(df['sentiment'].str.strip("'").str.lower().unique())
        sentiment = sentiment_filter.value
        if sentiment and sentiment != 'All':
            filtered_df = filtered_df[filtered_df['sentiment'].str.strip("'").str.lower() == sentiment]

        label_val = label_filter.value
        if label_val and label_val != 'All':
            filtered_df = filtered_df[filtered_df['label'].fillna('').apply(lambda x: label_val in [s.strip() for s in x.split(',')])]

        query = keyword_input.value.strip()
        if query:
            include_words = []
            exclude_words = []
            phrases = []
            tokens = query.split(' ')
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

            def match_keywords(text):
                text = text.lower()
                for word in exclude_words:
                    if word in text:
                        return False
                for word in include_words:
                    if word not in text:
                        return False
                for phrase in phrases:
                    if phrase not in text:
                        return False
                return True

            mask = filtered_df['title'].fillna('').apply(match_keywords) | filtered_df['body'].fillna('').apply(match_keywords)
            filtered_df = filtered_df[mask]

        grouped = filtered_df.groupby('title').agg(
            Article=('title', 'count'),
            Sentiment=('sentiment', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
            Link=('url', lambda x: x.dropna().iloc[0] if not x.dropna().empty else '-')
        ).reset_index().sort_values(by='Article', ascending=False)

        sentiments = filtered_df['sentiment'].str.strip("'").str.lower()
        total_artikel = filtered_df.shape[0]
        total_positif = (sentiments == 'positive').sum()
        total_negatif = (sentiments == 'negative').sum()
        total_netral = (sentiments == 'neutral').sum()

        display(HTML(f"""
        <b>Total Artikel:</b> {total_artikel} |
        <b>Positif:</b> {total_positif} |
        <b>Negatif:</b> {total_negatif} |
        <b>Netral:</b> {total_netral}<br><br>
        """))

        def format_sentiment(s):
            s = s.lower()
            if s == 'positive': return f'<span class="sentiment-positive">{s}</span>'
            if s == 'negative': return f'<span class="sentiment-negative">{s}</span>'
            return f'<span class="sentiment-neutral">{s}</span>'

        grouped['Sentiment'] = grouped['Sentiment'].apply(format_sentiment)
        grouped['Link'] = grouped['Link'].apply(lambda x: f'<a href="{x}" target="_blank">Lihat</a>' if x != '-' else '-')
        grouped['title'] = grouped['title'].apply(lambda x: f'<div title="{x}">{x}</div>')

        display(HTML(grouped.to_html(escape=False, index=False)))

# Layout upload dan tombol
input_layout = widgets.VBox([
    widgets.HBox([upload_widget, upload_button]),
    widgets.HBox([link_input, link_button]),
    widgets.HBox([sentiment_filter, keyword_input, label_filter]),
    status_output,
    output_area
])

display(input_layout)
apply_filters()