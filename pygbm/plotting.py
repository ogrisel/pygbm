from io import BytesIO

from graphviz import Digraph
import matplotlib.image as image
import matplotlib.pyplot as plt


def plot_tree(est_pygbm, est_lightgbm=None, tree_index=0, show=True):
    """
    Plot the i'th tree from an estimator. Can also plot a LightGBM estimator
    on the right for comparison.

    Requires matplotlib and graphviz (both python package and binary program).
    """
    def plot_pygbm(ax):
        """Plot the pygbm estimator on the given ax. Inspired from LightGBM
        code."""
        def make_graph():
            def add(node_idx, parent=None, decision=None):
                node = predictor_tree.nodes[node_idx]
                if node['is_leaf']:
                    name = 'leaf{0}'.format(node_idx)
                    label = 'leaf_index: {0}'.format(node_idx)
                    label += r'\nleaf_value: {0}'.format(node['value'])
                    label += r'\nleaf_count: {0}'.format(node['count'])
                    graph.node(name, label=label)
                else:
                    name = 'split{0}'.format(node_idx)
                    label = 'split_feature_index: {0}'.format(
                        node['feature_idx'])
                    label += r'\nthreshold: {0}'.format(node['threshold'])
                    for info in ('gain', 'value', 'count'):
                        label += r'\n{0}: {1}'.format(info, node[info])
                    graph.node(name, label=label)
                    add(node['left'], name, decision='<=')
                    add(node['right'], name, decision='>')

                if parent is not None:
                    graph.edge(parent, name, decision)

            graph = Digraph()
            add(0)  # add root to graph and recursively add child nodes
            return graph

        graph = make_graph()
        s = BytesIO()
        s.write(graph.pipe(format='png'))
        s.seek(0)
        img = image.imread(s)
        ax.imshow(img)

    ncols = 1 if est_lightgbm is None else 2
    fig, axes = plt.subplots(ncols=ncols, squeeze=False)  # axes is always 2D

    # plot pygbm tree
    predictor_tree = est_pygbm.predictors_[tree_index]
    plot_pygbm(axes[0][0])

    # plot lightgbm tree
    if est_lightgbm is not None:
        import lightgbm as lb
        lb.plot_tree(est_lightgbm, ax=axes[0][1], tree_index=tree_index,
                     show_info=['split_gain', 'internal_value',
                                'internal_count', 'leaf_count'])

    for ax in axes[0]:
        ax.axis('off')

    if show:
        plt.tight_layout()
        plt.show()
    return fig, axes