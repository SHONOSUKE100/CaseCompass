
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from transformers import AutoTokenizer, BertModel
import MeCab
from voyager import Index, Space
from gensim.models import doc2vec
import torch
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from dotenv import load_dotenv
import os
import sqlite3
import time


# Load environment variables
load_dotenv()



#データベース設定
dbname = 'case_compass.db'
conn = sqlite3.connect(dbname)

cur = conn.cursor()

# テーブル作成

cur.execute('''
CREATE TABLE IF NOT EXISTS summaries (
    case_id TEXT PRIMARY KEY,
    summary_text TEXT,
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# 変更をコミットし、接続を閉じる
conn.commit()
conn.close()

def insert_summary(case_id, summary_text):
    conn = sqlite3.connect('case_compass.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO summaries (case_id, summary_text) VALUES (?, ?)
    ''', (case_id, summary_text))
    conn.commit()
    conn.close()

def get_summary(case_id):
    conn = sqlite3.connect('case_compass.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT summary_text FROM summaries WHERE case_id = ?
    ''', (case_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None


# GeminiAPI設定
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="判例検索エンジン", layout="wide")

# Setup for generative AI model
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 800,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Helper functions
def load_file(file_path):
    """Loads JSON or TXT files based on extension."""
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            return json.load(file)
        elif file_path.endswith('.txt'):
            return file.read()

def to_markdown(text):
    """Converts text to Markdown format, specifically for Streamlit display."""
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def generate_response(case_name):
    """Generates a legal analysis response using the Google Generative AI model."""
    txt_file_path = f"precedents/text_files/{case_name}.txt"
    document = load_file(txt_file_path)

    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    prompt = [
    f"判例内容:{document}\n\nこの判例における主要な法律的論点を明らかにし、なぜこの論点が重要であるかを説明してください。",
    ]
    
    # Make the API call to generate content
    response = model.generate_content(prompt, stream=True)
    
    # Ensure the response completes iteration by calling resolve()
    response.resolve()  # This waits for all data to be fetched
    
    # Convert the response to Markdown and then display it in Streamlit
    response_text = to_markdown(response.text)

    return response_text


def display_results(clean_dataset, results):
    """Displays search results in the Streamlit app."""
    st.header('検索結果')

    #初期化
    if 'generated_text' not in st.session_state or 'placeholders' not in st.session_state:
        st.session_state['placeholders'] = {}
        st.session_state['generated_text'] = {}

    # 各判例について処理
    for id in results:
        # stateにidがない場合、初期設定
        if id not in st.session_state['placeholders']:
            st.session_state['placeholders'][id] = st.empty()
            st.session_state['generated_text'][id] = {"Bool": False, "Text": ""} 

    for id in results:
        with st.session_state['placeholders'][id].container():
            case = clean_dataset[id]
            st.subheader(case["case_name"])
            if 'gist' in case:
                st.write(f"**概要**: {case['gist']}")
            if 'date' in case:
                date = case['date']
                year = date["year"]
                month = date["month"]
                day = date["day"]
                
                st.write(f"**裁判日**: {year}年{month}月{day}日") 
            st.write(f"**裁判所ウェブサイト**: [こちらから]({case['detail_page_link']})")
            if st.button("要約を生成", key=f"generate_summary_{id}"):
                existing_summary = get_summary(id)
                st.session_state['generated_text'][id]["Bool"] = True
                if existing_summary:
                    generated_text = existing_summary
                else:
                    time.sleep(1)
                    generated_text = generate_response(id)
                    insert_summary(id, generated_text)

                st.session_state['generated_text'][id]["Text"] =generated_text
            if st.session_state['generated_text'][id]["Bool"]:
                st.warning("以下の要約はAIによって生成されたものです。AIは間違いを犯す可能性があります。必ず、上記のリンクから裁判所の裁判例集を確認してください。\n また、本サービスが提供する情報について一切責任を負いません。 \nまた、生成された文章は法的な意見を提供するものではありません。", icon="⚠️")
                st.markdown(st.session_state['generated_text'][id]["Text"], unsafe_allow_html=True)
            st.markdown("---")


# Search functions
def search_with_bow_unigram(query, clean_dataset):
    """Performs search using TF-IDF vectorizer."""
    vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
    query_vector = vectorizer.transform([MeCab.Tagger("-Owakati").parse(query)])
    index = Index.load("model/tf_idf_index.voy")
    neighbors, _ = index.query(query_vector.toarray().astype(np.float32), k=5)
    display_results(clean_dataset, [list(clean_dataset.keys())[n] for n in neighbors[0]])

def search_with_doc2vec(query, clean_dataset):
    """Performs search using Doc2Vec model."""
    model = doc2vec.Doc2Vec.load("model/doc2vec.model")
    similar_docs = model.dv.most_similar(positive=[model.infer_vector(query.split())], topn=5)
    display_results(clean_dataset, [list(clean_dataset.keys())[s[0]] for s in similar_docs])

def search_with_bert(query, clean_dataset):
    tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-whole-word-masking")
    model = BertModel.from_pretrained("SHONOSUKE/Addtional_Trained_BERT_For_Legal_Domain_v1")
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    doc_vector = outputs.last_hidden_state[:, 0, :].squeeze()

    index = Index.load("model/BERT.voy")
    doc_vector_array = doc_vector.cpu().detach().numpy()
    neighbors, distances = index.query(doc_vector_array, k=5)
    display_results(clean_dataset, [list(clean_dataset.keys())[n] for n in neighbors])

# def search_with_bow_unigram_for_all_documents(query, all_dataset):
#         """Performs search using TF-IDF vectorizer."""
#         vectorizer = joblib.load("model/tfidf_for_document_dataset_mindf_0.001_size_8791.joblib")
#         query_vector = vectorizer.transform([MeCab.Tagger("-Owakati").parse(query)])
#         index = Index.load("model/tf_idf_index_for_doc_mindf_0.001_size_8791.voy")
#         neighbors, _ = index.query(query_vector.toarray().astype(np.float32), k=5)
#         display_results(all_dataset, [list(all_dataset.keys())[n] for n in neighbors[0]])

# Main function to run the app
def main():
    st.title('判例検索エンジン')
    
    st.markdown("このアプリでは、簡単にかつ柔軟に判例を検索することができます。")

    # Choose search engine
    search_engine = st.radio('検索エンジンの種類を選択してください', ['TF-idf', "doc2vec", "BERT"], index=0)
    
    # Input search query
    query = st.text_area('検索したいワードを入力してください', '')

    # Load dataset
    clean_dataset = load_file('clean_list.json')

    # clean_dataset_all_documents = load_file('list_all.json')

    # Execute search
    if query:
        if search_engine == 'TF-idf':
            st.info("TF-idfで検索中です...")
            search_with_bow_unigram(query, clean_dataset)
        elif search_engine == 'doc2vec':
            st.info("Doc2Vecで検索中です...")
            search_with_doc2vec(query, clean_dataset)
        elif search_engine == 'BERT':
            st.info("BERTで検索中です...")
            search_with_bert(query, clean_dataset)
        else:
            st.error("選択された検索エンジンが無効です。")

        # elif search_engine == 'tfidf_alldoc':
        #     search_with_bow_unigram_for_all_documents(query, clean_dataset_all_documents)

if __name__ == "__main__":
    main()



