# clustering_pso_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------- PSO Algorithm -----------------

def PSO(problem, MaxIter=100, PopSize=100, c1=1.4962, c2=1.4962, w=0.7298, wdamp=1.0):
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'best_position': None,
        'best_cost': None,
    }

    CostFunction = problem['CostFunction']
    VarMin = problem['VarMin']
    VarMax = problem['VarMax']
    nVar = problem['nVar']

    gbest = {'position': None, 'cost': np.inf}

    pop = []
    for i in range(PopSize):
        pop.append(empty_particle.copy())
        pop[i]['position'] = np.random.uniform(VarMin, VarMax, nVar)
        pop[i]['velocity'] = np.zeros(nVar)
        pop[i]['cost'] = CostFunction(pop[i]['position'])
        pop[i]['best_position'] = pop[i]['position'].copy()
        pop[i]['best_cost'] = pop[i]['cost']

        if pop[i]['best_cost'] < gbest['cost']:
            gbest['position'] = pop[i]['best_position'].copy()
            gbest['cost'] = pop[i]['best_cost']

    for it in range(MaxIter):
        for i in range(PopSize):
            pop[i]['velocity'] = (
                w * pop[i]['velocity']
                + c1 * np.random.rand(nVar) * (pop[i]['best_position'] - pop[i]['position'])
                + c2 * np.random.rand(nVar) * (gbest['position'] - pop[i]['position'])
            )

            pop[i]['position'] += pop[i]['velocity']
            pop[i]['position'] = np.maximum(pop[i]['position'], VarMin)
            pop[i]['position'] = np.minimum(pop[i]['position'], VarMax)

            pop[i]['cost'] = CostFunction(pop[i]['position'])

            if pop[i]['cost'] < pop[i]['best_cost']:
                pop[i]['best_position'] = pop[i]['position'].copy()
                pop[i]['best_cost'] = pop[i]['cost']

                if pop[i]['best_cost'] < gbest['cost']:
                    gbest['position'] = pop[i]['best_position'].copy()
                    gbest['cost'] = pop[i]['best_cost']

        w *= wdamp

    return gbest, pop

# ----------------- Clustering Cost Function -----------------

def clustering_cost(centroids_flat, data, n_clusters):
    centroids = centroids_flat.reshape(n_clusters, -1)
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    cost = np.sum(min_distances)
    return cost

# ----------------- Assign Clusters -----------------

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return labels

# ----------------- Streamlit App -----------------

st.title('PSO Clustering App 🐦✨')

# Generate or upload data
st.sidebar.header("Data Settings")
n_samples = st.sidebar.slider('Number of Samples', 50, 500, 200)
n_features = 2  # Fixed at 2D for visualization

data_option = st.sidebar.radio('Data Option', ('Generate Random Data', 'Upload CSV File'))

if data_option == 'Generate Random Data':
    data = np.random.randn(n_samples, n_features)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        data = df.iloc[:, :2].values  # Use first two columns
    else:
        st.warning('Please upload a CSV file.')
        st.stop()

# PSO Parameters
st.sidebar.header("PSO Settings")
n_clusters = st.sidebar.slider('Number of Clusters', 2, 10, 3)
pop_size = st.sidebar.slider('Population Size', 10, 100, 30)
max_iter = st.sidebar.slider('Max Iterations', 10, 500, 100)

if st.button('Run PSO Clustering 🚀'):

    # Define problem
    problem = {
        'CostFunction': lambda x: clustering_cost(x, data, n_clusters),
        'nVar': n_clusters * n_features,
        'VarMin': np.min(data) - 1,
        'VarMax': np.max(data) + 1
    }

    # Run PSO
    gbest, _ = PSO(problem, MaxIter=max_iter, PopSize=pop_size)

    # Get final centroids
    best_centroids = gbest['position'].reshape(n_clusters, n_features)

    # Assign clusters
    labels = assign_clusters(data, best_centroids)

    # Plotting
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        cluster_data = data[labels == i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i+1}')
    ax.scatter(best_centroids[:, 0], best_centroids[:, 1], c='black', marker='X', s=100, label='Centroids')
    ax.set_title('PSO Clustering Result')
    ax.legend()
    st.pyplot(fig)

    st.success('Clustering Completed Successfully! 🎯')

