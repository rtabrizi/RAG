import streamlit as st
import torch
import numpy as np
import faiss
import PyPDF2
import os
import langchain

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, BartForQuestionAnswering
from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from streamlit import runtime

runtime.exists()

device = torch.device("cpu")
if torch.cuda.is_available():
   print("Training on GPU")
   device = torch.device("cuda:0")

file_url = "https://arxiv.org/pdf/1706.03762.pdf"
file_path = "assets/attention.pdf"

if not os.path.exists('assets'):
    os.mkdir('assets')

if not os.path.isfile(file_path):
    os.system(f'curl -o {file_path} {file_url}')
else:
    print("File already exists!")

class Retriever:

  def __init__(self, file_path, device, context_model_name, question_model_name):
    self.file_path = file_path
    self.device = device

    self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_name)
    self.context_model = DPRContextEncoder.from_pretrained(context_model_name).to(device)

    self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_model_name)
    self.question_model = DPRQuestionEncoder.from_pretrained(question_model_name).to(device)

  def token_len(self, text):
    tokens = self.context_tokenizer.encode(text)
    return len(tokens)

  def extract_text_from_pdf(self, file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

  def get_text(self):
    with open(self.file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

  def load_chunks(self):
    self.text = self.extract_text_from_pdf(self.file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=self.token_len,
        separators=["Section", "\n\n", "\n", ".", " ", ""]
    )

    self.chunks = text_splitter.split_text(self.text)

  def load_context_embeddings(self):
    encoded_input = self.context_tokenizer(self.chunks, return_tensors='pt', padding=True, truncation=True, max_length=150).to(device)

    with torch.no_grad():
      model_output = self.context_model(**encoded_input)
      self.token_embeddings = model_output.pooler_output.cpu().detach().numpy()

    self.index = faiss.IndexFlatL2(self.token_embeddings.shape[1])
    self.index.add(self.token_embeddings)

  def retrieve_top_k(self, query_prompt, k=10):
    encoded_query = self.question_tokenizer(query_prompt, return_tensors="pt", max_length=150, truncation=True, padding=True).to(device)

    with torch.no_grad():
        model_output = self.question_model(**encoded_query)
        query_vector = model_output.pooler_output

    query_vector_np = query_vector.cpu().numpy()
    D, I = self.index.search(query_vector_np, k)

    retrieved_texts = [' '.join(self.chunks[i].split('\n')) for i in I[0]]  # Replacing newlines with spaces

    return retrieved_texts

class RAG:
    def __init__(self,
                 file_path,
                 device,
                 context_model_name="facebook/dpr-ctx_encoder-multiset-base",
                 question_model_name="facebook/dpr-question_encoder-multiset-base",
                 generator_name="valhalla/bart-large-finetuned-squadv1"):

      # generator_name = "valhalla/bart-large-finetuned-squadv1"
      # generator_name = "'vblagoje/bart_lfqa'"
      # generator_name = "a-ware/bart-squadv2"

      generator_name = "valhalla/bart-large-finetuned-squadv1"
      self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_name)
      self.generator_model = BartForQuestionAnswering.from_pretrained(generator_name).to(device)

      self.retriever = Retriever(file_path, device, context_model_name, question_model_name)
      self.retriever.load_chunks()
      self.retriever.load_context_embeddings()


    def abstractive_query(self, question):
      self.generator_tokenizer = BartTokenizer.from_pretrained(self.generator_name)
      self.generator_model = BartForConditionalGeneration.from_pretrained(self.generator_name).to(device)
      context = self.retriever.retrieve_top_k(question, k=5)

      input_text = "answer: " + " ".join(context) + " " + question

      inputs = self.generator_tokenizer.encode(input_text, return_tensors='pt', max_length=150, truncation=True).to(device)
      outputs = self.generator_model.generate(inputs, max_length=150, min_length=2, length_penalty=2.0, num_beams=4, early_stopping=True)

      answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
      return answer

    def extractive_query(self, question):
      context = self.retriever.retrieve_top_k(question, k=15)
      
      inputs = self.generator_tokenizer(question, ". ".join(context), return_tensors="pt", truncation=True, max_length=150, padding="max_length")
      with torch.no_grad():
        model_inputs = inputs.to(device)
        outputs = self.generator_model(**model_inputs)

      answer_start_index = outputs.start_logits.argmax()
      answer_end_index = outputs.end_logits.argmax()

      if answer_end_index < answer_start_index:
        answer_start_index, answer_end_index = answer_end_index, answer_start_index

      predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
      answer = self.generator_tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
      answer = answer.replace('\n', ' ').strip()
      answer = answer.replace('$', '')

      return answer

context_model_name="facebook/dpr-ctx_encoder-single-nq-base"
question_model_name = "facebook/dpr-question_encoder-single-nq-base"

rag = RAG(file_path, device)

st.title("RAG Model Query Interface Chatbot")

# Initialize session state to keep track of the list of answers and questions
if 'history' not in st.session_state:
    st.session_state['history'] = []

question = st.text_input("Enter your question:")

if st.button("Ask"):
    # Fetch the answer for the question
    answer = rag.extractive_query(question)
    
    # Add the question and its answer to the history
    st.session_state.history.append({"type": "question", "content": question})
    st.session_state.history.append({"type": "answer", "content": answer})

# Display the chat history
for item in st.session_state.history:
    if item["type"] == "question":
        st.write(f"🧑 You: {item['content']}")
    else:
        st.write(f"🤖 Bot: {item['content']}")