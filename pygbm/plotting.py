from graphviz import Digraph

from pygbm import GradientBoostingMachine
import pygbm


def plot_tree(est_or_grower, est_lightgbm=None, tree_index=0, view=True, **kwargs):
    """Plot the i'th predictor tree of a GradientBoostingMachine instance,
    or the tree of a TreeGrower. Can also plot a LightGBM estimator (on the
    left) for comparison.

    est_or_grower can either be a GradientBoostingMachine instance or a
    TreeGrower. In this case tree_index is ignored, and more debugging info are
    displayed.

    Requires matplotlib and graphviz (both python package and binary program).

    kwargs are passed to graphviz.Digraph()

    Example:
    plotting.plot_tree(est_pygbm, est_lightgbm, view=False, filename='output')
    will silently save output to output.pdf
    """
    def make_pygbm_tree():
        def add_predictor_node(node_idx, parent=None, decision=None):
            predictor_tree = est_or_grower.predictors_[tree_index]
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
                add_predictor_node(node['left'], name, decision='<=')
                add_predictor_node(node['right'], name, decision='>')

            if parent is not None:
                graph.edge(parent, name, decision)

        def add_grower_node(node, parent=None, decision=None):
            name = 'split__{0}'.format(id(node))
            si = node.split_info
            if si is None:
                feature_idx = 0
                bin_idx = 0
                gain = 0.
                sum_gradients = 0.
                sum_hessians = 0.
            else:
                feature_idx = si.feature_idx
                gain = 0. if si.gain is None else si.gain
                bin_idx = si.bin_idx
                sum_gradients = si.gradient_left + si.gradient_right
                sum_hessians = si.hessian_left + si.hessian_right

            value = 0. if node.value is None else node.value
            label = '\nsplit_feature_index: {0}'.format(feature_idx)
            label += r'\nbin threshold: {:.4f}'.format(bin_idx)
            label += r'\ngain: {:.4f}'.format(gain)
            label += r'\nvalue: {:.4f}'.format(value)
            label += r'\ncount: {}'.format(node.sample_indices.shape[0])
            label += '\nuse_sub: {0}'.format(node.hist_subtraction)
            label += r'\nsum_gradients: {:.4f}'.format(sum_gradients)
            label += r'\nsum_hessians: {:.4f}'.format(sum_hessians)
            label += r'\nfind_split_time: {:.4f}'.format(node.find_split_time)
            label += r'\ncstr_speed: {:,} x 1e3'.format(
                int(node.construction_speed // 1e3))
            label += r'\napply_split_time: {:.4f}'.format(
                node.apply_split_time)

            graph.node(name, label=label)
            if node.value is None:  # not a leaf node
                add_grower_node(node.left_child, name, decision='<=')
                add_grower_node(node.right_child, name, decision='>')

            if parent is not None:
                graph.edge(parent, name, decision)

        if isinstance(est_or_grower, GradientBoostingMachine):
            add_predictor_node(0)
        elif isinstance(est_or_grower, pygbm.grower.TreeGrower):
            add_grower_node(est_or_grower.root)

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
