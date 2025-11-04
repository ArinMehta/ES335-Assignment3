import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="MLP Word Prediction", layout="wide", page_icon="🔮")

# Title
st.title("🔮 MLP Next-Word Prediction App")
st.markdown("**Interactive demo of trained MLP models for word prediction and embedding visualization**")

# Model definition (same as training)
def create_mlp_model(vocab_size, embedding_dim, hidden_size=1024, context_length=5, activation='relu'):
    class MLPWordPredictor(nn.Module):
        def __init__(self):
            super(MLPWordPredictor, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            input_size = context_length * embedding_dim
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
            self.fc2 = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x):
            embedded = self.embedding(x)
            embedded = embedded.reshape(embedded.size(0), -1)
            hidden = self.activation(self.fc1(embedded))
            output = self.fc2(hidden)
            return output
        
        def get_embeddings(self):
            return self.embedding.weight.detach().cpu().numpy()
    
    return MLPWordPredictor()

# Cache data loading
@st.cache_resource
def load_data():
    # Load datasets
    with open('wikipedia_processed.pkl', 'rb') as f:
        wiki_data = pickle.load(f)
    with open('linux_processed.pkl', 'rb') as f:
        linux_data = pickle.load(f)
    
    return {
        'wiki': {
            'vocab_size': wiki_data['vocab_size'],
            'word_to_idx': wiki_data['word_to_idx'],
            'idx_to_word': wiki_data['idx_to_word']
        },
        'linux': {
            'vocab_size': linux_data['vocab_size'],
            'word_to_idx': linux_data['word_to_idx'],
            'idx_to_word': linux_data['idx_to_word']
        }
    }

# Cache model loading
@st.cache_resource
def load_model(model_name, dataset_info):
    embedding_dim = 64 if '64d' in model_name else 32
    activation = 'tanh' if 'tanh' in model_name else 'relu'
    vocab_size = dataset_info['vocab_size']
    
    model = create_mlp_model(vocab_size, embedding_dim, 1024, 5, activation)
    ckpt = torch.load(f"{model_name}_best.pth", map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    return model, ckpt.get('val_loss', 0), ckpt.get('val_accuracy', 0)

# Load data once
data = load_data()

# Sidebar
st.sidebar.header("⚙️ Model Selection")
dataset_choice = st.sidebar.selectbox("Dataset", ["Wikipedia", "Linux Kernel"])
dataset_key = 'wiki' if dataset_choice == "Wikipedia" else 'linux'
dataset_prefix = 'wiki' if dataset_choice == "Wikipedia" else 'linux'

model_variant = st.sidebar.selectbox(
    "Model Configuration",
    ["64d-ReLU", "32d-ReLU", "64d-tanh", "32d-tanh"]
)

model_name = f"{dataset_prefix}_{model_variant}"
dataset_info = data[dataset_key]

# Load selected model
try:
    model, val_loss, val_acc = load_model(model_name, dataset_info)
    st.sidebar.success(f"✅ Model loaded: {model_name}")
    st.sidebar.metric("Validation Loss", f"{val_loss:.4f}")
    st.sidebar.metric("Validation Accuracy", f"{val_acc:.2f}%")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["🔮 Word Prediction", "📊 Embedding Visualization"])

# Tab 1: Word Prediction
with tab1:
    st.header("Next-Word Prediction")
    st.markdown("Enter 5 consecutive words and the model will predict the next word.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_input(
            "Enter 5 words (space-separated):",
            placeholder="e.g., the quick brown fox jumps",
            key="prediction_input"
        )
    
    with col2:
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    if predict_btn and input_text:
        words = input_text.strip().lower().split()
        
        if len(words) != 5:
            st.error("⚠️ Please enter exactly 5 words.")
        else:
            # Check if words are in vocabulary
            word_to_idx = dataset_info['word_to_idx']
            idx_to_word = dataset_info['idx_to_word']
            unk_token = '<unk>'
            
            indices = []
            for w in words:
                if w in word_to_idx:
                    indices.append(word_to_idx[w])
                else:
                    indices.append(word_to_idx.get(unk_token, 0))
                    st.warning(f"Word '{w}' not in vocabulary, using <unk>")
            
            # Predict
            with torch.no_grad():
                context = torch.LongTensor([indices])
                output = model(context)
                probs = torch.softmax(output, dim=1)
                top_k = torch.topk(probs, k=10, dim=1)
                
            st.success("**Prediction Results:**")
            
            # Display top predictions
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                st.subheader("Top 10 Predictions")
                for i, (prob, idx) in enumerate(zip(top_k.values[0], top_k.indices[0]), 1):
                    word = idx_to_word[int(idx)]
                    st.write(f"**{i}.** {word} — {prob.item()*100:.2f}%")
            
            with col_b:
                st.subheader("Probability Distribution (Top 10)")
                top_words = [idx_to_word[int(idx)] for idx in top_k.indices[0]]
                top_probs = [prob.item()*100 for prob in top_k.values[0]]
                
                fig = px.bar(
                    x=top_probs, 
                    y=top_words, 
                    orientation='h',
                    labels={'x': 'Probability (%)', 'y': 'Word'},
                    color=top_probs,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Embedding Visualization
with tab2:
    st.header("Word Embedding Visualization")
    st.markdown("Visualize learned word embeddings in 2D space using PCA or t-SNE.")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_words = st.text_input(
            "Enter words to visualize (comma-separated):",
            placeholder="e.g., king, queen, man, woman, computer",
            key="viz_input"
        )
    
    with col2:
        method = st.selectbox("Reduction Method", ["PCA", "t-SNE"])
    
    with col3:
        viz_btn = st.button("📊 Visualize", type="primary", use_container_width=True)
    
    if viz_btn and search_words:
        words_list = [w.strip().lower() for w in search_words.split(',') if w.strip()]
        
        if len(words_list) < 2:
            st.error("⚠️ Please enter at least 2 words.")
        else:
            word_to_idx = dataset_info['word_to_idx']
            idx_to_word = dataset_info['idx_to_word']
            
            # Get embeddings
            embeddings = model.get_embeddings()
            
            # Filter valid words
            valid_words = []
            valid_indices = []
            for w in words_list:
                if w in word_to_idx:
                    valid_words.append(w)
                    valid_indices.append(word_to_idx[w])
                else:
                    st.warning(f"Word '{w}' not in vocabulary")
            
            if len(valid_words) < 2:
                st.error("⚠️ Not enough valid words in vocabulary.")
            else:
                word_embeddings = embeddings[valid_indices]
                
                # Dimensionality reduction
                if method == "PCA":
                    reducer = PCA(n_components=2)
                else:
                    reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(valid_words)-1))
                
                embeddings_2d = reducer.fit_transform(word_embeddings)
                
                # Plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[:, 0],
                    y=embeddings_2d[:, 1],
                    mode='markers+text',
                    text=valid_words,
                    textposition='top center',
                    marker=dict(size=12, color=np.arange(len(valid_words)), colorscale='Viridis'),
                    textfont=dict(size=12)
                ))
                
                fig.update_layout(
                    title=f"Word Embeddings ({method})",
                    xaxis_title=f"{method} Component 1",
                    yaxis_title=f"{method} Component 2",
                    height=600,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Visualized {len(valid_words)} words using {method}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**MLP Word Prediction Demo**")
st.sidebar.markdown("Built with PyTorch & Streamlit")