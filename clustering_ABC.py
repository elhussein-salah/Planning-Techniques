import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ----------------------------
# 🔹 1. Data Generation
# ----------------------------
@st.cache_data
def generate_data(n_samples=300, n_clusters=3, n_features=2):
    data, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)
    return data

# ----------------------------
# 🔹 2. Fitness Calculation
# ----------------------------
def calculate_fitness(centers, data):
    distances = np.min([np.linalg.norm(data - c, axis=1) for c in centers], axis=0)
    return np.sum(distances)

def assign_clusters(data, centers):
    distances = [np.linalg.norm(data - c, axis=1) for c in centers]
    return np.argmin(distances, axis=0)

def update_solution(solution, phi):
    rand_index = np.random.randint(len(solution))
    new_solution = solution + phi * (solution - solution[rand_index])
    return new_solution

# ----------------------------
# 🔹 3. ABC Algorithm
# ----------------------------
def artificial_bee_colony(data, n_clusters=3, max_iter=100, n_bees=20, limit=20, progress=None):
    dim = data.shape[1]
    food_sources = [np.random.uniform(np.min(data), np.max(data), size=(n_clusters, dim)) for _ in range(n_bees)]
    fitness = [calculate_fitness(fs, data) for fs in food_sources]
    trial = [0] * n_bees

    for iteration in range(max_iter):
        # Employed Bees
        for i in range(n_bees):
            phi = np.random.uniform(-1, 1, size=food_sources[i].shape)
            new_solution = update_solution(food_sources[i], phi)
            new_fitness = calculate_fitness(new_solution, data)
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Onlooker Bees
        prob = fitness / np.sum(fitness)
        for _ in range(n_bees):
            i = np.random.choice(range(n_bees), p=(1 - prob) / np.sum(1 - prob))
            phi = np.random.uniform(-1, 1, size=food_sources[i].shape)
            new_solution = update_solution(food_sources[i], phi)
            new_fitness = calculate_fitness(new_solution, data)
            if new_fitness < fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Scout Bees
        for i in range(n_bees):
            if trial[i] > limit:
                food_sources[i] = np.random.uniform(np.min(data), np.max(data), size=(n_clusters, dim))
                fitness[i] = calculate_fitness(food_sources[i], data)
                trial[i] = 0

        if progress:
            progress.progress((iteration + 1) / max_iter)

    best_index = np.argmin(fitness)
    return food_sources[best_index], fitness[best_index]

# ----------------------------
# 🔹 4. Plotting Clusters
# ----------------------------
def plot_clusters(data, labels, centers):
    fig, ax = plt.subplots()
    for i in np.unique(labels):
        ax.scatter(data[labels == i][:, 0], data[labels == i][:, 1], label=f"Cluster {i}")
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Centers')
    ax.set_title("Clustering Result (ABC Algorithm)")
    ax.legend()
    return fig

# ----------------------------
# 🔹 5. Streamlit UI
# ----------------------------
st.set_page_config(page_title="ABC Clustering", layout="centered")
st.title("🐝 Clustering using Artificial Bee Colony Algorithm")

# User inputs
n_samples = st.slider("Number of data points:", 100, 1000, 300, step=50)
n_clusters = st.slider("Number of clusters:", 2, 10, 3)
n_iter = st.slider("Max iterations:", 10, 300, 100, step=10)
n_bees = st.slider("Number of bees:", 5, 50, 20)
limit = st.slider("Limit before scout phase:", 5, 50, 20)

# Run the algorithm
if st.button("Run ABC Clustering"):
    st.write("⏳ Running Artificial Bee Colony Algorithm...")
    data = generate_data(n_samples=n_samples, n_clusters=n_clusters)
    progress_bar = st.progress(0)
    centers, best_fitness = artificial_bee_colony(
        data, n_clusters=n_clusters, max_iter=n_iter, n_bees=n_bees, limit=limit, progress=progress_bar
    )
    labels = assign_clusters(data, centers)
    fig = plot_clusters(data, labels, centers)
    st.pyplot(fig)
    st.success(f"✅ Done! Best fitness: {best_fitness:.2f}")
