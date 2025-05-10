# 📌 Project Title: Clustering with ABC & PSO and Path Optimization using ACO

## 🧠 Description

This project combines **swarm intelligence algorithms** to perform intelligent data clustering and optimal pathfinding. It is divided into two main phases:

---

### 🔹 Phase 1: Data Clustering with ABC and PSO

In this phase, we use:

- **Artificial Bee Colony (ABC)**
- **Particle Swarm Optimization (PSO)**

to perform **unsupervised clustering** of data points. Each algorithm independently searches for an optimal cluster configuration by minimizing intra-cluster distance and maximizing inter-cluster separation. The goal is to compare both techniques and select the most effective for a given dataset.

---

### 🔹 Phase 2: Best Path Finding using Ant Colony Optimization (ACO)

After clustering, **Ant Colony Optimization (ACO)** is employed to find the most efficient path (shortest or least-cost path) through the centroids of the clusters. Inspired by real ant behavior, ACO simulates pheromone-based learning to iteratively improve the path over time.

---

## ✅ Key Objectives

- Implement **ABC** and **PSO** for robust clustering of multidimensional data.
- Evaluate and compare clustering performance using metrics like **SSE**, **Silhouette Score**, etc.
- Use **ACO** to solve a **Travelling Salesman Problem (TSP)**-like path between cluster centers.
- Visualize the clustering and optimal path.

---

## 🧰 Technologies

- **Python**
  - `NumPy`
  - `matplotlib`
  - `scikit-learn`
- Custom implementations of:
  - **ABC (Artificial Bee Colony)**
  - **PSO (Particle Swarm Optimization)**
  - **ACO (Ant Colony Optimization)**
