import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')


    def render(self, mode='rgb_array'):
        fig = plt.figure(figsize=(20, 20))
        ax = fig.gca()
        ax.axis('off')

        pos = nx.get_node_attributes(self.world, 'coords')
        G = nx.DiGraph(self.world)
        nodelist = []
        edgelist = []
        action = self.last_action_for_drawing
        act = self.itEnv.action_space_shape[0]
        node_colors = []
        edge_colors = []
        for n in self.world.nodes():
            node_action = action[act*n:act*(n+1)]
            nodelist.append(n)
            node_colors.append(node_action[-1])
            j = 0
            added = 0
            for nn in self.world.neighbors(n):
                if node_action[j] > 0:
                    edgelist.append((n,nn))
                    edge_colors.append(node_action[j])
                    added += 1
                j += 1
            assert abs(np.sum(node_action) - 1) < 0.00001, node_action
            assert node_action[-1] != 0 or added > 0, (node_action, n)

        nx.draw_networkx(G, edgelist=edgelist, edge_color=edge_colors, vmin=-1, vmax=1, node_shape='.', edge_vmax=1.1,
                            cmap=matplotlib.cm.get_cmap("Blues"), edge_cmap=matplotlib.cm.get_cmap("Blues"),
                            node_color=node_colors, nodelist=nodelist, pos=pos, arrows=True, with_labels=False, ax=ax)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        # Option 2a: Convert to a NumPy array
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        plt.close(fig)
        return X

    def render(self, mode='rgb_array'):
        fig = plt.figure()
        ax = fig.gca()
        ax.axis('off')

        x = np.zeros((2, self.world_size))
        for i in range(self.world_size):
            x[0, i] = self.world.nodes[i]['coords'][0]
            x[1, i] = self.world.nodes[i]['coords'][1]

        plt.scatter(x[0,:], x[1,:])
        #plt.arrow(self.position, 0, self.last_move*0.5, 0, length_includes_head=True, head_width=0.003, head_length=0.2) #x,y,dx,dy

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        # Option 2a: Convert to a NumPy array
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        plt.close(fig)
        return X