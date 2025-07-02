import streamlit as st
import fitz  # PyMuPDF
import networkx as nx
from pyvis.network import Network
from collections import Counter
import tempfile
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# === NEW: NLTK for stop word removal ===
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

st.set_page_config(layout="wide")
st.title("📄 Multi-File Word Graph Visualizer with Community Clustering")

# === Helper functions ===

def extract_words_from_pdf(file, use_nltk=True):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    words = []
    for page in doc:
        words += page.get_text("words")
    doc.close()
    if use_nltk:
        return [w[4].lower() for w in words if w[4].isalpha() and w[4].lower() not in stop_words]
    else:
        return [w[4].lower() for w in words if w[4].isalpha()]

def extract_words_from_txt(file, use_nltk=True):
    text = file.read().decode("utf-8")
    if use_nltk:
        return [w.lower() for w in text.split() if w.isalpha() and w.lower() not in stop_words]
    else:
        return [w.lower() for w in text.split() if w.isalpha()]

def build_word_graph(words, window_size=2):
    G = nx.Graph()
    for i in range(len(words) - 1):
        for j in range(1, min(window_size + 1, len(words) - i)):
            word1, word2 = words[i], words[i + j]
            if G.has_edge(word1, word2):
                G[word1][word2]["weight"] += 1
            else:
                G.add_edge(word1, word2, weight=1)
    return G

def interpolate_color_colormap(val, maxval, colormap_name="plasma"):
    cmap = cm.get_cmap(colormap_name)
    return mcolors.to_hex(cmap(val / (maxval + 1e-6)))

def edge_color_colormap(weight, max_weight, colormap_name="coolwarm"):
    cmap = cm.get_cmap(colormap_name)
    return mcolors.to_hex(cmap(weight / (max_weight + 1e-6)))

def compute_node_embeddings(G):
    pca = PCA(n_components=2)
    adj_matrix = nx.to_numpy_array(G)
    pos = pca.fit_transform(adj_matrix)
    return {node: pos[i] for i, node in enumerate(G.nodes())}

def draw_clusters_convex_hulls(G, embeddings, word_freq, num_clusters=4):
    pos_array = np.array(list(embeddings.values()))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pos_array)
    labels = kmeans.labels_
    cluster_nodes = {i: [] for i in range(num_clusters)}
    for i, node in enumerate(G.nodes()):
        cluster_nodes[labels[i]].append(node)

    fig = go.Figure()
    colors = px.colors.qualitative.Set3

    for cluster_idx, nodes in cluster_nodes.items():
        cluster_pos = np.array([embeddings[node] for node in nodes])
        x, y = cluster_pos[:, 0], cluster_pos[:, 1]
        size = [word_freq[node] for node in nodes]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(size=[10 + 30 * (s / max(size)) for s in size], color=colors[cluster_idx % len(colors)], opacity=0.8),
            text=nodes,
            name=f"Cluster {cluster_idx + 1}",
            textposition='middle center'
        ))

        if len(cluster_pos) >= 3:
            hull = ConvexHull(cluster_pos)
            hull_pts = cluster_pos[hull.vertices]
            hull_pts = np.append(hull_pts, [hull_pts[0]], axis=0)
            fig.add_trace(go.Scatter(
                x=hull_pts[:, 0], y=hull_pts[:, 1],
                fill='toself',
                mode='lines',
                line=dict(color=colors[cluster_idx % len(colors)], width=2),
                name=f"Hull {cluster_idx + 1}",
                opacity=0.3
            ))

    fig.update_layout(
        title="🔶 Clustered Word Network with Convex Hulls",
        showlegend=True,
        height=700,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# === File uploader ===

uploaded_files = st.file_uploader("📂 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    window_size = st.slider("🔗 Co-occurrence Window Size", 1, 10, 2)
    top_n = st.slider("🔝 Top N Words/Nodes per Graph", 10, 100, 30)

    # === Buttons to select NLTK filtering ===
    with_nltk = st.button("Use NLTK Stopword Filtering")
    without_nltk = st.button("Use No Stopword Filtering")

    if with_nltk or without_nltk:
        for file in uploaded_files:
            with st.spinner(f"🔍 Processing {file.name}..."):
                words = extract_words_from_pdf(file, use_nltk=with_nltk) if file.name.endswith(".pdf") else extract_words_from_txt(file, use_nltk=with_nltk)
                word_freq = Counter(words)
                G = build_word_graph(words, window_size=window_size)
                top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]
                G_small = G.subgraph([node for node, _ in top_nodes])

            st.subheader(f"🌐 Word Network for **{file.name}**")
            net = Network(height="600px", width="100%", bgcolor="#222", font_color="white")

            max_freq = max(word_freq.values())
            max_edge_weight = max([d['weight'] for _, _, d in G_small.edges(data=True)], default=1)

            for node in G_small.nodes():
                freq = word_freq.get(node, 1)
                size = 10 + (50 - 10) * (freq / max_freq)
                color = interpolate_color_colormap(freq, max_freq)
                net.add_node(node, label=node, title=f"{node}: {freq}", size=size, color=color)

            for u, v, d in G_small.edges(data=True):
                color = edge_color_colormap(d["weight"], max_edge_weight)
                net.add_edge(u, v, value=d["weight"], color=color)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                net.save_graph(tmp_file.name)
                tmp_file_path = tmp_file.name

            st.components.v1.html(open(tmp_file_path).read(), height=600)
            os.remove(tmp_file_path)

            st.subheader(f"📊 Top {top_n} Words in {file.name}")
            df_top_words = pd.DataFrame(word_freq.most_common(top_n), columns=["Word", "Frequency"])
            st.plotly_chart(px.bar(df_top_words, x="Word", y="Frequency", color="Frequency", color_continuous_scale="Plasma"), use_container_width=True)

            st.subheader(f"🫧 Bubble Chart with Labels and Ranks for {file.name}")
            bubble_data = []
            total = sum([freq for _, freq in word_freq.most_common(top_n)])
            for rank, (word, freq) in enumerate(word_freq.most_common(top_n), 1):
                perc = round((freq / total) * 100, 2)
                bubble_data.append({"Word": word, "Frequency": freq, "Percentage": perc, "Rank": f"{rank}ᵗʰ" if rank > 3 else ["1st", "2nd", "3rd"][rank-1]})

            df_bubbles = pd.DataFrame(bubble_data)
            df_bubbles["Label"] = df_bubbles["Rank"] + "<br>" + df_bubbles["Word"] + "<br>" + df_bubbles["Percentage"].astype(str) + "%"

            fig_bubble = px.scatter(df_bubbles, x="Frequency", y="Percentage", size="Percentage", text="Label", color="Rank", size_max=100)
            fig_bubble.update_traces(textposition='middle center')
            fig_bubble.update_layout(height=700, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig_bubble, use_container_width=True)

            st.subheader(f"🧩 Icon Array Pictograph for Top {top_n} Words")
            icon = "🔵"
            max_icons = 10
            pictograph = []
            for word, freq in word_freq.most_common(top_n):
                count = max(1, int((freq / max_freq) * max_icons))
                pictograph.append({"Word": word, "Icons": [icon] * count, "Y": list(range(1, count + 1))})

            fig_icons = go.Figure()
            for row in pictograph:
                fig_icons.add_trace(go.Scatter(x=[row["Word"]] * len(row["Icons"]), y=row["Y"], mode="text", text=row["Icons"], textfont=dict(size=18), showlegend=False))

            fig_icons.update_layout(title="Icon Array", xaxis_title="Word", yaxis=dict(showticklabels=False), height=400)
            st.plotly_chart(fig_icons, use_container_width=True)

            st.subheader(f"🔥 Word Co-occurrence Heatmap for {file.name}")
            matrix = pd.DataFrame(0, index=G_small.nodes(), columns=G_small.nodes())
            for u, v, d in G_small.edges(data=True):
                matrix.at[u, v] = d["weight"]
                matrix.at[v, u] = d["weight"]
            heatmap = px.imshow(matrix, labels=dict(x="Word", y="Word", color="Co-occurrence"), color_continuous_scale="Plasma")
            st.plotly_chart(heatmap, use_container_width=True)

            st.subheader(f"📋 Word Co-occurrence Matrix for {file.name}")
            filtered = matrix.loc[[n for n, _ in top_nodes], [n for n, _ in top_nodes]]
            st.dataframe(filtered.style.background_gradient(cmap="Purples").format(precision=0))
            st.download_button("⬇️ Download Matrix CSV", data=filtered.to_csv().encode("utf-8"), file_name=f"{file.name}_matrix.csv", mime="text/csv")

            st.subheader(f"🧭 4-Quadrant Word Matrix for {file.name}")
            centrality = nx.degree_centrality(G)
            matrix_df = pd.DataFrame([{"Word": word, "Frequency": freq, "Centrality": centrality.get(word, 0)} for word, freq in word_freq.most_common(top_n)])
            fig_quad = px.scatter(matrix_df, x="Frequency", y="Centrality", text="Word", color="Frequency", color_continuous_scale="Plasma", title="4-Quadrant Matrix")
            fig_quad.add_vline(x=matrix_df["Frequency"].mean(), line_dash="dash", line_color="gray")
            fig_quad.add_hline(y=matrix_df["Centrality"].mean(), line_dash="dash", line_color="gray")
            st.plotly_chart(fig_quad, use_container_width=True)

            st.subheader(f"🎯 Circular Histogram of Word Occurrence for {file.name}")
            polar_words = word_freq.most_common(top_n)
            angles = np.linspace(0, 2 * np.pi, len(polar_words), endpoint=False)
            fig_circ = go.Figure()
            fig_circ.add_trace(go.Barpolar(r=[f for _, f in polar_words], theta=np.degrees(angles), text=[w for w, _ in polar_words], marker_color=[f for _, f in polar_words], marker_colorscale='Plasma'))
            fig_circ.update_layout(polar=dict(radialaxis=dict(showticklabels=True)), showlegend=False)
            st.plotly_chart(fig_circ, use_container_width=True)

            st.subheader(f"🧠 Community Clustered Network with Convex Hulls for {file.name}")
            embeddings = compute_node_embeddings(G_small)
            fig_clustered = draw_clusters_convex_hulls(G_small, embeddings, word_freq, num_clusters=4)
            st.plotly_chart(fig_clustered, use_container_width=True)
