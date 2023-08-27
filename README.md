# Retrieval-Augmented Generation
### Take-Home Project

This is my RAG implementation for PDF question answering. The PDF is the original *Attention Is All You Need* paper and the questions are based on architectural and design choices for the paper.

### Use
This project was developed almost exclusively in Google CoLab but was adapted to Streamlit. The colab notebook can be found [here](https://colab.research.google.com/drive/1RnoNBC__hs8fzz6-vCfxBH13Jxl2A5a8?usp=sharing) and is self-contained, meaning you can run it from top to bottom to observe functionality.

For 

### Installation

Complete the following steps:
1. clone the repository to your desired location using `git clone https://github.com/rtabrizi/RAG.git`
   
2. Navigate to the project directory

3. Set up a virtual environment to manage dependencies:
    `python -m venv venv`
    
4. Activate the virtual environment: `.\venv\Scripts\activate`

5. Install the requirements: `pip install -r requirements.txt
`

Run the Streamlit app: `pip install -r requirements.txt`

---

## Somewhat Successful Outputs
 
Question: What is the self-attention mechanism also known as? \
Answer: intra-attention

Question: What is another name for self-attention? \
 Answer: intra-attention

Question: Can you explain the Transformer architecture to me in simple terms? \
 Answer: relying entirely on an attention mechanism to draw global dependencies between input and output

Question: How is the self-attention mechanism different from other attention mechanisms? \
 Answer: relating different positions of a single sequence in order to compute a representation of the sequence

Question: In what ways does self-attention improve model performance? \
 Answer: more interpretable models

Question: What's the purpose behind using self-attention in the Transformer? \
 Answer: draw global dependencies between input and output

Question: What are the main components of the Transformer architecture? \
 Answer: attention mechanism to draw global dependencies between input and output

Question: Can you outline the advantages of the Transformer model over RNNs? \
 Answer: The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality

## Failure Cases

Question: What are the benefits of using multiple attention heads? \ Answer: operations, albeit at the cost of reduced

Question: What is the benefit of multi-head attention? \
 Answer: operations, albeit at the cost of reduced

Question: What problem does multi-head attention solve in the Transformer architecture? \
 Answer: draw global dependencies between input and output

 Question: Why do we use multiple attention heads? \
 Answer: positions, an effect we counteract with Multi-Head Attention as described in section 3.2. Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].. just - this is what we are missing, in my opinion. <EOS> Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in an


Question: How does the Transformer model use position encodings? \
 Answer: draw global dependencies between input and output

Question: How does attention mechanism handle sequence order? \
 Answer: relating different positions of a single sequence

Question: Why are positional encodings crucial in Transformers? \
 Answer: positions [ 12]. In the Transformer this

Question: Describe the role of key-value pairs in the attention mechanism. \
 Answer: Describe the role of key-value pairs in the attention mechanism.self-attention and discuss its advantages over models such as [17, 18] and [9]. 3 Model Architecture Most competitive neural sequence transduction models have an encoder-decoder structure [ 5,2,35]. Here, the encoder maps an input sequence of symbol representations (x1,..., x n)to a sequence of continuous representations z= (z1,..., z n). Given z, the decoder then generates an output. reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2. Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].. attention over the output of the encoder stack
