![RAG (1)](https://github.com/rtabrizi/RAG/assets/30701017/a150dd01-1d24-4d2d-a63f-0e232a9fe44c)
# Take-Home Project

This is my RAG implementation for PDF question answering. The PDF is the original *Attention Is All You Need* paper and the questions are based on architectural and design choices for the paper.

## Usage
This project was developed almost exclusively in Google Colab on an A100 GPU but was adapted to Streamlit. The colab notebook can be found [here](https://colab.research.google.com/drive/1RnoNBC__hs8fzz6-vCfxBH13Jxl2A5a8?usp=sharing) and is self-contained, meaning you can run it from top to bottom to observe functionality.
The streamlit app is very finnicky and I must host it on my end on huggingface spaces. Outputs between this and the colab are not consistent and I suspect there is an indexing issue going on that is causing these incoherent outputs.

## Design Choices and Component Descriptions

### Retriever
The retriever is its own python class that is saved as an attribute in any RAG instance. 

**LangChain Recursive Character Text Splitter** \
This text splitter was LangChain's endorsed splitter. I adapted their docs example to use HuggingFace embeddings through the use of a context encoder `"facebook/dpr-ctx_encoder-single-nq-base"`. This question encoder was trained on the Natural Questions (NQ) dataset. This is a BERT-based encoder simply trained on the NQ dataset. This encoder's token outputs are used to determine the chunking length of the text since RecursiveCharacterTextSplitter operates with the token lengths, not the original text lengths. We allow for some overlap to avoid any relevant information being split separately across two chunks.

**Similarity Search: FAISS** \
All the chunks are stored in a list attribute in the retriever object. We then use a DPR question encoder `"facebook/dpr-question_encoder-single-nq-base"` to get the question tokens. This question encoder, also BERT-based, was trained on the NQ dataset as well. Facebook AI Similarity Search then performs an optimized KNN on the chunks and query/question tokens with an L2 distance. We return these top k chunks to the RAG and append them to the back of the original questions string.

### RAG
The RAG is represented as a python class whose main components are the generator model and tokenizer saved as attributes, as well as a retriever object saved as an attribute.

**Generator** \
I use a BARTForQuestionAnswering instance to generate the answers upon getting the updated prompt and context from the retriever. BARTForQuestionAnswering is an extractive QA model in that it simply extracts the indices corresponding to the highest logits based on the prompt. As such, the model is incapable of generating text truly on its own and instead extracts content from the context and, in some cases, the original question itself. In the reflection below, I discuss why this design choice was made.

### Reflection, More Design Choices

There were many issues that surfaced in creating this RAG.

#### Data Preprocessing
The attention paper, as many know, is filled with convoluted figures and benchmark tables that a PDF parser like PyPDF2 simply can't meaningfully capture. For instance, what most humans takeaway from the paper is figure 1 showing the transformer. No matter how I phrase my query, our RAG will not be able to extract this information without some sophisticated approaches like OCR. I did my best to split on newlines and spaces, as well as remove any special characters, so that the extracted context chunks are meaningful and coherent. That said, many issues still arose like determining the best context length that captured most relevant information while limiting the amount of noisy text per chunk. 

#### Adaptive Answer Generation

I had originally attempted and envisioned an adaptive answer generator in which the context chunks simply aided the BART model. This BART model was BARTForConditionalGeneration. What I observed was that the model produced nonsensical results that simply didn't suffice as answers (think: complete gibberish). There were many parameters to consider for this behavior, such as temperature or the number of beams used in search, and I determined it was not in the best interest of time to pursue this. That said, this would've been the most 'human-like' RAG generator, not to mention that the RAG was designed with the intent of grounding parametric memory in language models with non-parametric references like a PDF.

#### Extractive Answer Generation
Instead of an adaptive generator, I opt for an extractive answer generator which exclusively pulls from the provided PDF. In doing so, we can ensure that the model doesn't halluciante facts from its parametric memory but we suffer from incoherent answers as shown below. This involved using BARTForQuestionAnswering whose tokenizer had designated question and context parameters. Instead of generating answers from parametric memory conditioned on the nonparametric memory, this model learns to decode the prompt-engineered input and returns the indices corresponding to the max logits of the result. In other words, we simply index into the text (extract) instead of generating text conditioned on the context (adaptive generation). Of course, the chunks were never perfectly spliced, so the answers extracted from these chunks can be ungrammatical and incoherent as well. As a solution, I added support to only output complete sentences (by extracting the tokens with max logits and then iterating until the nearest EOS). I ended up not including this implementation as I felt these answers were not as concise and direct as I'd like them to be.

#### Retriever Revisited
I left a colab cell that displays what the retrieved context is based on a particular query. I observed that the context that a human expert would choose is often retrieved but isn't necessarily the top neighbor/chunk or even in the top 5 closest chunks. I had originally opted to use the `"facebook/dpr-ctx_encoder-multiset-base"` context encoder and switched to the NQ encoder and observed improvement in this regard. This yielded some improved retrieval but there is still plenty of room for improvement, including trying even more context and question encoders and altering the chunk length.

#### Prompt Engineering
Because I ultimately opted for an extractive answer generator, I was aware that the queries I provided would have to be similar in structure and language to the actual text from the paper. You can see my attempt in the first 3 queries in the **Failure Cases** section down below. These 3 prompts are largely similar but exhibit different results. 

## Installation

In order to use basic terminal UI for attention paper QA. main.py will create assets/attention.pdf if you don't already have the appropriate directory and pdf.

Complete the following steps:
1. clone the repository to your desired location using `git clone https://github.com/rtabrizi/RAG.git`
   
2. Navigate to the project directory

3. Set up a virtual environment to manage dependencies:
    `python -m venv venv`
    
4. Activate the virtual environment: `source venv/bin/activate`

5. Install the requirements: `pip install -r requirements.txt`

6. Run the script: `python main.py`

---

## Somewhat Successful Outputs
 
Question: What is the self-attention mechanism also known as? \
Answer: intra-attention

Question: What is another name for self-attention? \
 Answer: intra-attention

Question: What is the purpose of multi-head attention? \
Answer: allows the model to jointly attend to information from different representation subspaces at different positions

Question: In what ways does self-attention improve model performance? \
 Answer: more interpretable models

Question: What's the purpose behind using self-attention in the Transformer? \
 Answer: efficiently handle large inputs and outputs

Question: What are the main components of the Transformer architecture? \
 Answer: stacked self-attention and point-wise, fully connected layers for both the encoder and decoder

Question: Can you outline the advantages of the Transformer model over RNNs? \
 Answer: significantly more parallelization


## Failure Cases
**Note** Intentionally exhaustive list. I feel it's much more meaningful to reveal these failure cases than ignoring them. In my opinion, they are more important than the successful outputs. 

Question: Can you explain the Transformer architecture to me in simple terms? \
 Answer: olutions. The Transformer follows this overall architecture using stacked

Question: What are the benefits of using multiple attention heads? \
Answer: operations, albeit at the cost of reduced

Question: How is the self-attention mechanism different from other attention mechanisms? \
 Answer: relating different positions

Question: What is the benefit of multi-head attention? \
 Answer: operations, albeit at the cost of reduced

Question: is this trained on GPU or CPU? \
GPUs

Question: what happens when context length increases? \
reduced effective resolution

Question: What problem does multi-head attention solve in the Transformer architecture? \
 Answer: draw global dependencies between input and output

 Question: why do we use multiple attention heads? \
Answer: positions, an effect we counteract with Multi-Head Attention as. mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions. Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been. during training. 4 Why Self-Attention In this section we compare various aspects of self-attention layers to the recurrent and convolu-. subspaces at different positions. With a single attention head, averaging inhibits this. MultiHead( Q, K, V ) = Concat(head 1,...,head h)WO. different layer types. As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially. opinion. <EOS> Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top:. to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.. depicted in Figure 2. Multi-head attention allows

Question: How does the Transformer model use position encodings? \
 Answer: stacks. For the base model, we use a rate of Pdrop= 0.1.. it more difficult to learn dependencies between distant positions [ 12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due. mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions. The Transformer follows this overall architecture using

Question: How does attention mechanism handle sequence order? \
 Answer: we must inject some information

Question: Why are positional encodings crucial in Transformers? \
 Answer: stacks. For the base model, we use a rate of Pdrop= 0.1.. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,. 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the. encoder. •Similarly, self

Question: Describe the role of key-value pairs in the attention mechanism. \
 Answer: mapping a query and a set of key-value pairs to an output

 Question: what is the difference between an encoder and decoder? \
self-attention layers in the decoder allow each position in the decoder to attend to. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,. efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].. typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].. and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the. the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes. entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly. [38, 2, 9]. •The encoder contains self-attention layers. In a self-attention layer all of the keys, values. sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head. respectively. 3.1 Encoder and Decoder Stacks Encoder: The encoder is composed of a stack of N= 6 identical layers



