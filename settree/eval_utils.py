import json
from pathlib import Path
import textwrap
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,  average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve, matthews_corrcoef
)


def save_roc_curves(
    curves: Mapping[str, Tuple[Sequence[int], Sequence[float], float]],
    output_path: Path,
    title: str = "ROC Curve",
) -> Path:
    """Save ROC curves from aligned binary labels and positive-class scores.

    Each entry maps a legend name to (labels, scores, already-computed AUC).
    Scores must be continuous evaluation probabilities, not thresholded labels.
    The caller supplies an existing result directory. Only the figure is written.
    """
    output_path = Path(output_path)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }):
        figure, axis = plt.subplots(figsize=(6, 6), facecolor="white")
        try:
            colors = ("#0072B2", "#D55E00")
            for index, (name, (labels, scores, auc)) in enumerate(curves.items()):
                fpr, tpr, _ = roc_curve(labels, scores)
                axis.plot(
                    fpr, tpr,
                    color=colors[index % len(colors)],
                    linewidth=2,
                    linestyle="-" if index % 2 == 0 else "--",
                    label=f"{name} (AUC = {auc:.3f})",
                )
            axis.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
            axis.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title=title,
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                facecolor="white",
            )
            axis.set_axisbelow(True)
            axis.grid(color="0.9", linewidth=0.6)
            axis.legend(loc="lower right", frameon=False)
            figure.tight_layout()
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(figure)
    return output_path


def _thesis_tree_label(node: dict) -> str:
    """Format an exported binary probability tree node for display only."""
    if node["type"] == "leaf":
        return (
            f"Patients = {node['num_patients']:,}\n"
            f"P(positive) = {float(node['prediction']):.3f}"
        )
    offset = node["offset"]
    if offset is None:
        window = "Full timeline"
    else:
        anchor = str(node["anchor"])
        if anchor.endswith(" Anchor"):
            anchor = anchor[:-len(" Anchor")]
        # Match anchor_utils: integer offsets are calendar months; fractions
        # are rounded signed 30-day multiples, not fractional calendar months.
        offset = float(offset)
        unit = "months" if offset.is_integer() else "days"
        extent = int(offset) if offset.is_integer() else round(offset * 30)
        interval = f"0–{extent}" if extent >= 0 else f"−{abs(extent)} to 0"
        window = f"{anchor}: {interval} {unit}"
    feature = textwrap.fill(str(node["feature"]), width=25)
    window = textwrap.fill(window, width=28)
    return (
        f"{feature}\n{window}\n"
        f"{str(node['aggregation']).capitalize()} ≤ {node['threshold']:.6g}"
    )


def _load_thesis_tree(path: Path) -> tuple:
    """Read exported nodes and lay out leaves without altering the tree."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {node["node_id"]: node for node in records}
    if not nodes or len(nodes) != len(records):
        raise ValueError(f"Empty tree or duplicate node IDs in {path}.")
    child_ids = {
        node[key]
        for node in records if node["type"] == "split"
        for key in ("left_child", "right_child")
    }
    roots = set(nodes) - child_ids
    if len(roots) != 1 or not child_ids.issubset(nodes):
        raise ValueError(f"Tree must have one root and complete children: {path}")
    positions, labels, visited = {}, {}, set()
    leaf_count = 0

    def visit(node_id, depth):
        nonlocal leaf_count
        if node_id in visited:
            raise ValueError(f"Cycle or shared child in {path}.")
        visited.add(node_id)
        node = nodes[node_id]
        if node["type"] == "leaf":
            probability = float(node["prediction"])
            if not 0 <= probability <= 1:
                raise ValueError(f"Expected binary leaf probability in {path}.")
            x = leaf_count
            leaf_count += 1
        elif node["type"] == "split":
            left = visit(node["left_child"], depth + 1)
            right = visit(node["right_child"], depth + 1)
            x = (left + right) / 2
        else:
            raise ValueError(f"Unknown node type in {path}: {node['type']}")
        labels[node_id] = _thesis_tree_label(node)
        positions[node_id] = (x, -depth)
        return x

    visit(next(iter(roots)), 0)
    if visited != set(nodes):
        raise ValueError(f"Unreachable nodes in {path}.")
    return nodes, positions, labels, leaf_count


def _draw_thesis_tree(axis, tree: tuple, title: str) -> None:
    """Draw one tree with the same spacing, colors, and typography per panel."""
    nodes, positions, labels, leaf_count = tree
    boxes = {}
    for node_id, node in nodes.items():
        boxes[node_id] = axis.text(
            *positions[node_id], labels[node_id],
            ha="center", va="center", fontsize=10, linespacing=1.4,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#EAF3F8" if node["type"] == "split" else "#EDF4E9",
                "edgecolor": "#455A64", "linewidth": 0.9,
            },
            zorder=3,
        )
    axis.set_xlim(-0.65, max(leaf_count - 0.35, 0.65))
    axis.set_ylim(min(y for _, y in positions.values()) - 0.55, 0.65)
    axis.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=18)
    axis.axis("off")
    # Resolve box sizes before clipping each edge to the node borders.
    axis.figure.canvas.draw()
    for node_id, node in nodes.items():
        if node["type"] != "split":
            continue
        start = positions[node_id]
        for child_key, branch in (("left_child", "≤"), ("right_child", ">")):
            child_id = node[child_key]
            end = positions[child_id]
            axis.add_patch(FancyArrowPatch(
                start, end,
                patchA=boxes[node_id].get_bbox_patch(),
                patchB=boxes[child_id].get_bbox_patch(),
                arrowstyle="-", color="#66757F", linewidth=1.1,
                shrinkA=3, shrinkB=3, zorder=1,
            ))
            axis.text(
                (start[0] + end[0]) / 2, (start[1] + end[1]) / 2,
                branch, ha="center", va="center", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
                zorder=2,
            )


def save_thesis_tree_figures(
    tree_paths: Mapping[str, Path], output_dir: Path,
) -> dict:
    """Save individual and stacked FA/GI figures from exported tree JSON only.

    Intended for Experiment 1's scalar binary probability leaves. Leaf counts
    are training patients. Thresholds use six significant digits and leaf
    probabilities three decimals for display; source artifacts are untouched.
    """
    trees = {
        outcome: _load_thesis_tree(tree_paths[outcome])
        for outcome in ("fa", "gi")
    }
    width = max(10, max(tree[3] for tree in trees.values()) * 1.9)
    heights = {
        outcome: max(3.5, 2 + 1.45 * max(-y for _, y in tree[1].values()))
        for outcome, tree in trees.items()
    }
    paths = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False}):
        for outcome, tree in trees.items():
            figure, axis = plt.subplots(figsize=(width, heights[outcome]))
            try:
                figure.subplots_adjust(
                    left=0.03, right=0.97, bottom=0.07, top=0.88,
                )
                _draw_thesis_tree(axis, tree, f"{outcome.upper()} learned tree")
                paths[outcome] = (
                    Path(tree_paths[outcome]).parent / f"{outcome}_tree_thesis.png"
                )
                figure.savefig(paths[outcome], dpi=300, facecolor="white")
            finally:
                plt.close(figure)
        figure, axes = plt.subplots(
            2, 1, figsize=(width, sum(heights.values())),
            gridspec_kw={"height_ratios": list(heights.values())},
        )
        try:
            figure.subplots_adjust(
                left=0.03, right=0.97, bottom=0.035, top=0.94, hspace=0.3,
            )
            for axis, (outcome, tree), panel in zip(
                axes, trees.items(), ("(a)", "(b)"),
            ):
                _draw_thesis_tree(axis, tree, f"{panel} {outcome.upper()} learned tree")
            paths["combined"] = Path(output_dir) / "learned_trees_fa_vs_gi_thesis.png"
            figure.savefig(paths["combined"], dpi=300, facecolor="white")
        finally:
            plt.close(figure)
    return paths


def evaluate_TemporalTree(y_real, y_predicted, y_probabilities=None):
        """
        Evaluate classification performance with metrics suitable for imbalanced data.
        
        Parameters:
        - y_real: Array-like, true labels (ground truth).
        - y_predicted: Array-like, predicted labels.
        - y_probabilities: Array-like, predicted probabilities for positive class (needed for AUC-ROC).

        Returns:
        - A dictionary with accuracy, precision, recall, F1 score, MCC, AUC-ROC, and confusion matrix.
        - Displays an AUC-ROC curve plot.
        """
        # Compute traditional metrics
        
        accuracy = accuracy_score(y_real, y_predicted)
        precision = precision_score(y_real, y_predicted, average='binary', zero_division=0)
        recall = recall_score(y_real, y_predicted, average='binary', zero_division=0)
        f1 = f1_score(y_real, y_predicted, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_real, y_predicted)  # More balanced metric for imbalanced datasets
        conf_matrix = confusion_matrix(y_real, y_predicted)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'confusion_matrix': conf_matrix
        }
        
        # AUC-ROC Calculation & Plot
        if y_probabilities is not None:
            auc_roc = roc_auc_score(y_real, y_probabilities)
            fpr, tpr, _ = roc_curve(y_real, y_probabilities)
            print(f'len(fpr): {len(fpr)}')
            # Store AUC in results
            results['auc_roc'] = auc_roc
            # Average Precision / PR-AUC
            average_precision = average_precision_score(
                y_real,
                y_probabilities
            )
            results['average_precision'] = average_precision

            pr_precision, pr_recall, _ = precision_recall_curve(
                y_real,
                y_probabilities
            )
            # Plot ROC Curve
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='blue', label=f'AUC-ROC = {auc_roc:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid()
            plt.show()
        return results

def plot_tree_structure(tree, figsize=(12, 6), title="Temporal Decision Tree", save_path=None):
    """
    Plots a simple visual representation of the tree structure using matplotlib.

    Parameters:
        tree (TemporalSplitNode or TemporalLeaf): The root node of the trained tree.
        figsize (tuple): Size of the plot.
        title (str): Title of the tree.
    """
    fig, ax = plt.subplots(figsize=figsize)
    y_spacing = 1.5

    def _plot_node(node, x=0, y=0, dx=6, depth=0):
        label = node.describe() if hasattr(node, "describe") else str(node)
        ax.text(x, -y, label, ha='center', bbox=dict(
            facecolor='lightblue' if not node.is_leaf() else 'lightgreen',
            edgecolor='black',
            boxstyle='round,pad=0.4'
        ))

        if not node.is_leaf():
            # Draw left child
            if node.left:
                x_left = x - dx / (depth + 1)
                y_left = y + y_spacing
                ax.plot([x, x_left], [-y, -y_left], 'k-')
                _plot_node(node.left, x_left, y_left, dx, depth + 1)

            # Draw right child
            if node.right:
                x_right = x + dx / (depth + 1)
                y_right = y + y_spacing
                ax.plot([x, x_right], [-y, -y_right], 'k-')
                _plot_node(node.right, x_right, y_right, dx, depth + 1)

    _plot_node(tree)
    ax.axis('off')
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
