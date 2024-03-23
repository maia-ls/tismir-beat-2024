import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import defaultdict

def fast_votek(embeddings, select_num, k, rho=10):
    """Select samples using vote-k

    Args:
        embeddings (np.ndarray): Embeddings of samples (Nfiles, Ndims)
        select_num (int): Number of files to select
        k (int): Number of neighbors to consider
        rho (scalar): Scoring parameter

    Returns:
        selected_indices (array): Indices of selected samples
    """
    # This code was adapted from the following repo:
    # https://github.com/HKUNLP/icl-selective-annotation/blob/main/two_steps.py
    n = len(embeddings)

    bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    for i in range(n):
        cur_emb = embeddings[i].reshape(1, -1)
        cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
        sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
        for idx in sorted_indices:
            if idx!=i:
                vote_stat[idx].append(i)
        bar.update(1)

    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += rho ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices

def diversity(embeddings, select_num):
    """Select samples using diversity strategy

    Args:
        embeddings (np.ndarray): Embeddings of samples (Nfiles, Ndims)
        select_num (int): Number of files to select

    Returns:
        selected_indices (array): Indices of selected samples
    """
    # This code was adapted from the following repo:
    # https://github.com/HKUNLP/icl-selective-annotation/blob/main/two_steps.py

    selected_indices = []
    first_id = random.choice(range(len(embeddings)))
    bar = tqdm(range(select_num),desc=f'voting')
    bar.update(1)
    selected_indices.append(first_id)
    selected_representations = embeddings[first_id].reshape(1, -1)
    for count in range(select_num - 1):
        scores = np.sum(cosine_similarity(embeddings, selected_representations), axis=1)
        for i in selected_indices:
            scores[i] = np.inf
        min_idx = np.argmin(scores)
        selected_representations = np.concatenate((selected_representations,
                                                   embeddings[min_idx].reshape(1, -1)), axis=0)
        selected_indices.append(min_idx)
        bar.update(1)
    return selected_indices

def mfl(embeddings, select_num):
    """Select samples using maximum facility location strategy

    Args:
        embeddings (np.ndarray): Embeddings of samples (Nfiles, Ndims)
        select_num (int): Number of files to select

    Returns:
        selected_indices (array): Indices of selected samples
    """
    # This code was adapted from the following repo:
    # https://github.com/HKUNLP/icl-selective-annotation/blob/main/two_steps.py

    N, D = embeddings.shape
    cosine = cosine_similarity(embeddings, embeddings)
    selected = np.zeros(N, dtype=bool)
    max_similarity = np.zeros(N) - 1
    for k in tqdm(range(select_num)):
        marginal_gain = np.sum(np.maximum(0, cosine - max_similarity), axis=1) * (1 - selected.astype(float))
        node = np.argmax(marginal_gain)
        selected[node] = True
        max_similarity = np.maximum(max_similarity, cosine[node])
    selected_indices = np.nonzero(selected)[0].tolist()
    return selected_indices