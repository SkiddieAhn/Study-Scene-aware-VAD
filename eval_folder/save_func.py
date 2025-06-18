import matplotlib.pyplot as plt
import numpy as np
import os
from fastprogress import progress_bar


def save_auc_result(cfg, save_dir, dataset_name, training_mode, iter_num, best_auc):
    auc_path = os.path.join(save_dir, f'{dataset_name}_auc_{training_mode}.txt')
    with open(auc_path, 'a') as f:
        f.write(f'[seed:{cfg.manualseed}] Iter {iter_num}: AUC = {best_auc:.4f}\n')


def save_score_auc_graph(answers_idx, scores, auc, file_path, x='Frame', y='Anomaly Score'):
    length = len(scores)
    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.plot([num for num in range(length)],[score for score in scores]) # plotting
    plt.plot([], [], ' ', label=f'AUC: {auc:.2f}')
    plt.bar(answers_idx, 1, width=1, color='green',alpha=0.5, label='Ground-truth') # check answer
    plt.ylim(0, 1)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(file_path)


def plot_tsne_eval(cfg, tsne_model, dataset_name, scene_feature_list, save_dir):
    # Flatten features and labels
    all_features, all_labels = [], []
    for s in range(cfg.scene_length):
        all_features.extend(scene_feature_list[s])
        all_labels.extend([s + 1] * len(scene_feature_list[s]))
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Define colors, markers, labels
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', '^', 'D']
    label_names = [f'scene{i+1}' for i in range(cfg.scene_length)]

    # t-SNE
    tsne_embedding = tsne_model.fit_transform(all_features)

    # Plotting
    plt.figure(figsize=(8, 6), dpi=150)
    for s in range(cfg.scene_length):
        idxs = (all_labels == s + 1)
        plt.scatter(tsne_embedding[idxs, 0], tsne_embedding[idxs, 1],
                    c=colors[s], marker=markers[s], label=label_names[s], s=30, alpha=0.7)
    plt.title(f'{dataset_name} t-SNE (normal features)')
    plt.legend()
    plt.tight_layout()

    # Save
    tsne_path = os.path.join(save_dir, f'tsne_{cfg.trained_model}.jpg')
    plt.savefig(tsne_path)
    plt.clf()


