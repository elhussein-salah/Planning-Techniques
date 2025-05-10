import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Map class
class Map:
    class Nodes:
        def __init__(self, row, col, in_map, spec):
            self.node_pos = (row, col)
            self.edges = self.compute_edges(in_map)
            self.spec = spec

        def compute_edges(self, map_arr):
            imax, jmax = map_arr.shape
            edges = []
            if map_arr[self.node_pos[0]][self.node_pos[1]] == 1:
                for dj in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        newi = self.node_pos[0] + di
                        newj = self.node_pos[1] + dj
                        if (dj == 0 and di == 0):
                            continue
                        if (0 <= newj < jmax) and (0 <= newi < imax):
                            if map_arr[newi][newj] == 1:
                                edges.append({'FinalNode': (newi, newj),
                                              'Pheromone': 1.0, 'Probability': 0.0})
            return edges

    def __init__(self, map_array):
        self.in_map = map_array
        self.occupancy_map = self._map_2_occupancy_map()
        self.initial_node = (int(np.where(self.in_map == 'S')[0][0]), int(np.where(self.in_map == 'S')[1][0]))
        self.final_node = (int(np.where(self.in_map == 'F')[0][0]), int(np.where(self.in_map == 'F')[1][0]))
        self.nodes_array = self._create_nodes()

    def _create_nodes(self):
        return [[self.Nodes(i, j, self.occupancy_map, self.in_map[i][j]) 
                 for j in range(self.in_map.shape[1])] 
                 for i in range(self.in_map.shape[0])]

    def _map_2_occupancy_map(self):
        map_arr = np.copy(self.in_map)
        map_arr[map_arr == 'O'] = 0
        map_arr[map_arr == 'E'] = 1
        map_arr[map_arr == 'S'] = 1
        map_arr[map_arr == 'F'] = 1
        return map_arr.astype(int)

    def represent_map(self):
        plt.figure(figsize=(5,5))
        plt.plot(self.initial_node[1], self.initial_node[0], 'ro', markersize=10)
        plt.plot(self.final_node[1], self.final_node[0], 'bo', markersize=10)
        plt.imshow(self.occupancy_map, cmap='gray', interpolation='nearest')
        st.pyplot(plt)
        plt.close()

    def represent_path(self, path):
        plt.figure(figsize=(5,5))
        x, y = zip(*[(p[1], p[0]) for p in path])
        plt.plot(x, y, 'r-')
        plt.plot(self.initial_node[1], self.initial_node[0], 'go', markersize=10)
        plt.plot(self.final_node[1], self.final_node[0], 'bo', markersize=10)
        plt.imshow(self.occupancy_map, cmap='gray', interpolation='nearest')
        st.pyplot(plt)
        plt.close()

# Ant Colony class
class AntColony:
    class Ant:
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node = start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []
            self.final_node_reached = False
            self.remember_visited_node(start_node_pos)

        def move_ant(self, node_to_visit):
            self.actual_node = node_to_visit
            self.remember_visited_node(node_to_visit)

        def remember_visited_node(self, node_pos):
            self.visited_nodes.append(node_pos)

        def get_visited_nodes(self):
            return self.visited_nodes

        def is_final_node_reached(self):
            if self.actual_node == self.final_node:
                self.final_node_reached = True

        def enable_start_new_path(self):
            self.final_node_reached = False

        def setup_ant(self):
            self.visited_nodes = [self.start_pos]
            self.actual_node = self.start_pos

    def __init__(self, in_map, no_ants, iterations, evaporation_factor, pheromone_adding_constant):
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.paths = []
        self.ants = self.create_ants()
        self.best_result = []

    def create_ants(self):
        return [self.Ant(self.map.initial_node, self.map.final_node) for _ in range(self.no_ants)]

    def select_next_node(self, actual_node):
        if not actual_node.edges:
            return None  # No edges to move

        total_sum = sum(edge['Pheromone'] for edge in actual_node.edges)
        edges_list = []
        probs = []
        for edge in actual_node.edges:
            prob = edge['Pheromone'] / total_sum
            edge['Probability'] = prob
            edges_list.append(edge)
            probs.append(prob)

        for edge in actual_node.edges:
            edge['Probability'] = 0.0

        return np.random.choice(edges_list, 1, p=probs)[0]['FinalNode']

    def pheromone_update(self):
        self.sort_paths()
        for path in self.paths:
            for j, element in enumerate(path):
                for edge in self.map.nodes_array[element[0]][element[1]].edges:
                    if (j+1) < len(path) and edge['FinalNode'] == path[j+1]:
                        edge['Pheromone'] = (1.0 - self.evaporation_factor) * edge['Pheromone'] + self.pheromone_adding_constant / float(len(path))
                    else:
                        edge['Pheromone'] = (1.0 - self.evaporation_factor) * edge['Pheromone']

    def empty_paths(self):
        self.paths = []

    def sort_paths(self):
        self.paths.sort(key=len)

    def add_to_path_results(self, in_path):
        self.paths.append(in_path)

    def get_coincidence_indices(self, lst, element):
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset + 1)
            except ValueError:
                return result
            result.append(offset)

    def delete_loops(self, in_path):
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            coincidences.reverse()
            for i, coincidence in enumerate(coincidences):
                if i != len(coincidences) - 1:
                    res_path[coincidences[i+1]:coincidence] = []
        return res_path

    def calculate_path(self):
        for i in range(self.iterations):
            for ant in self.ants:
                ant.setup_ant()
                while not ant.final_node_reached:
                    next_node = self.select_next_node(
                        self.map.nodes_array[ant.actual_node[0]][ant.actual_node[1]])
                    if next_node is None:
                        break  # No move possible, stuck
                    ant.move_ant(next_node)
                    ant.is_final_node_reached()

                self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
                ant.enable_start_new_path()

            if self.paths:
                self.pheromone_update()
                self.best_result = self.paths[0]
                st.write(f'Iteration: {i+1}, Length of path: {len(self.best_result)}')
            else:
                st.warning(f"Iteration {i+1}: No valid paths found.")
            self.empty_paths()

        return self.best_result

# Streamlit App
st.title("Ant Colony Optimization Path Planning 🐜")

uploaded_file = st.file_uploader("Upload a map file (.txt)", type=["txt"])

if uploaded_file is not None:
    map_array = np.loadtxt(uploaded_file, dtype=str)
    map_obj = Map(map_array)

    st.subheader("Map Preview:")
    map_obj.represent_map()

    ants = st.slider("Number of Ants", 1, 100, 10)
    iterations = st.slider("Number of Iterations", 1, 200, 50)
    p = st.slider("Evaporation Rate (p)", 0.0, 1.0, 0.5, step=0.05)
    Q = st.number_input("Pheromone Adding Constant (Q)", 0.1, 100.0, 10.0)

    if st.button("Start Optimization"):
        colony = AntColony(map_obj, ants, iterations, p, Q)
        path = colony.calculate_path()
        if path:
            st.success("Optimization Completed!")
            st.subheader("Best Path Found:")
            st.write(path)
            map_obj.represent_path(path)
        else:
            st.error("No path found. Try uploading a different map!")
