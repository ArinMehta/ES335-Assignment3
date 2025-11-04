import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="MLP Word Prediction", layout="wide", page_icon="🔮")

# Title
st.title("🔮 MLP Next-Word Prediction App with Temperature Control")
st.markdown("**Interactive demo with adjustable context length, embedding dimension, activation, temperature, and more**")

# Model definition
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

# Cache model loading with fixed naming
@st.cache_resource
def load_model(model_name, dataset_info, context_len):
    embedding_dim = 64 if '64d' in model_name else 32
    activation_str = 'tanh' if 'tanh' in model_name else 'relu'
    vocab_size = dataset_info['vocab_size']
    
    model = create_mlp_model(vocab_size, embedding_dim, 1024, context_len, activation_str)
    ckpt = torch.load(f"{model_name}_best.pth", map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    return model, ckpt.get('val_loss', 0), ckpt.get('val_accuracy', 0)

def sample_with_temperature(logits, temperature=1.0, random_state=None):
    logits = logits.cpu().numpy()
    if random_state is not None:
        np.random.seed(random_state)
    logits = logits / (temperature if temperature > 0 else 1e-8)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return np.random.choice(len(logits), p=probs), probs

# Load data once
data = load_data()

# Sidebar - Model Configuration
st.sidebar.header("⚙️ Model Configuration")

dataset_choice = st.sidebar.selectbox("Dataset", ["Wikipedia", "Linux Kernel"])
dataset_key = 'wiki' if dataset_choice == "Wikipedia" else 'linux'
dataset_prefix = 'wiki' if dataset_choice == "Wikipedia" else 'linux'

embedding_dim = st.sidebar.selectbox("Embedding Dimension", [32, 64], index=1)
activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh"], index=0)

# Match file naming convention: ReLU (uppercase) for relu, tanh (lowercase)
act_str = "ReLU" if activation == "relu" else "tanh"
model_name = f"{dataset_prefix}_{embedding_dim}d-{act_str}"

dataset_info = data[dataset_key]

st.sidebar.header("🎛️ Generation Parameters")
context_len = st.sidebar.slider("Context Length", 1, 10, 5)
k_words = st.sidebar.slider("Number of Words to Predict (k)", 1, 10, 3)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.05)
random_seed = st.sidebar.slider("Random Seed", 1, 1000, 42)

# Load selected model
try:
    model, val_loss, val_acc = load_model(model_name, dataset_info, context_len)
    st.sidebar.success(f"✅ Model: {model_name}")
    st.sidebar.metric("Val Loss", f"{val_loss:.4f}")
    st.sidebar.metric("Val Acc", f"{val_acc:.2f}%")
except Exception as e:
    st.sidebar.error(f"Error: {e}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["🔮 Word Prediction", "📊 Embedding Visualization"])

# Tab 1: Word Prediction with Temperature
with tab1:
    st.header("Next K-Word Prediction with Temperature Control")
    st.markdown(f"Enter at least **{context_len}** words and the model will predict the next **{k_words}** word(s).")
    
    input_text = st.text_input(
        f"Input Prompt (min {context_len} words):",
        placeholder="e.g., the quick brown fox jumps over the lazy dog",
        key="prediction_input"
    )
    
    predict_btn = st.button("🔮 Predict Next K Words", type="primary", use_container_width=True)
    
    if predict_btn and input_text:
        words = input_text.strip().lower().split()
        
        if len(words) < context_len:
            st.error(f"⚠️ Please enter at least {context_len} words.")
        else:
            word_to_idx = dataset_info['word_to_idx']
            idx_to_word = dataset_info['idx_to_word']
            unk = word_to_idx.get('<unk>', 0)
            
            # Use last context_len words as context
            ctx = [word_to_idx.get(w, unk) for w in words[-context_len:]]
            predictions = []
            all_probs = []
            
            # Generate k words
            for step in range(k_words):
                ctx_tensor = torch.LongTensor([ctx])
                with torch.no_grad():
                    out = model(ctx_tensor)[0]
                next_idx, probs = sample_with_temperature(out, temperature, random_seed + step)
                next_word = idx_to_word[next_idx]
                predictions.append(next_word)
                
                # Store top 10 probabilities for this step
                top_10_idx = np.argsort(probs)[-10:][::-1]
                all_probs.append({idx_to_word[i]: float(probs[i]) for i in top_10_idx})
                
                # Update context (sliding window)
                ctx = ctx[1:] + [next_idx] if context_len > 1 else [next_idx]
            
            st.success("**Prediction Results:**")
            
            # Display predicted words
            st.subheader("Predicted Words:")
            st.markdown(f"**{' '.join(predictions)}**")
            
            # Display step-by-step probabilities
            st.subheader("Step-by-Step Top 10 Probabilities:")
            
            for i, (word, step_probs) in enumerate(zip(predictions, all_probs), 1):
                with st.expander(f"Step {i}: \"{word}\""):
                    prob_df = {
                        "Word": list(step_probs.keys()),
                        "Probability (%)": [p * 100 for p in step_probs.values()]
                    }
                    
                    fig = px.bar(
                        prob_df,
                        x="Probability (%)",
                        y="Word",
                        orientation='h',
                        color="Probability (%)",
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Embedding Visualization
with tab2:
    st.header("Word Embedding Visualization")
    st.markdown("Visualize learned word embeddings in 2D space using PCA.")
    
    search_words = st.text_input(
        "Enter words to visualize (comma-separated):",
        placeholder="e.g., king, queen, man, woman, computer",
        key="viz_input"
    )
    
    viz_btn = st.button("📊 Visualize Embeddings", type="primary", use_container_width=True)
    
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
                
                # PCA reduction
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(word_embeddings)
                
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
                    title="Word Embeddings (PCA 2D)",
                    xaxis_title="PCA Component 1",
                    yaxis_title="PCA Component 2",
                    height=600,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✅ Visualized {len(valid_words)} words using PCA")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**MLP Word Prediction Demo**")
st.sidebar.markdown("Built with PyTorch & Streamlit")