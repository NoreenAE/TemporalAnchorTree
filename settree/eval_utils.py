import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,  average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve, matthews_corrcoef
)

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
