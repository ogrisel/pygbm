from graphviz import Digraph


def plot_tree(est_pygbm, est_lightgbm=None, tree_index=0, view=True, **kwargs):
    """Plot the i'th tree from an estimator. Can also plot a LightGBM estimator
    (on the left) for comparison.

    Requires matplotlib and graphviz (both python package and binary program).

    kwargs are passed to graphviz.Digraph()

    Example:
    plotting.plot_tree(est_pygbm, est_lightgbm, view=False, filename='output')
    will silently save output to output.pdf
    """
    def make_pygbm_tree():
        def add(node_idx, parent=None, decision=None):
            predictor_tree = est_pygbm.predictors_[tree_index]
            node = predictor_tree.nodes[node_idx]
            name = 'split__{0}'.format(node_idx)
            label = '\nsplit_feature_index: {0}'.format(
                node['feature_idx'])
            label += r'\nthreshold: {:.4f}'.format(node['threshold'])
            label += r'\ngain: {:.4f}'.format(node['gain'])
            label += r'\nvalue: {:.4f}'.format(node['value'])
            label += r'\ncount: {}'.format(node['count'])
            label += '\nuse_sub: {0}'.format(node['use_sub'])

            graph.node(name, label=label)
            if not node['is_leaf']:
                add(node['left'], name, decision='<=')
                add(node['right'], name, decision='>')

            if parent is not None:
                graph.edge(parent, name, decision)

        add(0)  # add root to graph and recursively add child nodes

    # make lightgbm tree
    if est_lightgbm is not None:
        import lightgbm as lb
        graph = lb.create_tree_digraph(
            est_lightgbm,
            tree_index=tree_index,
            show_info=['split_gain', 'internal_value', 'internal_count',
                       'leaf_count'],
            **kwargs)
    else:
        graph = Digraph(**kwargs)

    # make pygbm tree
    make_pygbm_tree()

    graph.render(view=view)
