from typing import List
import math
import numpy as np
import pandas as pd


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """
    Computes the Recall@k metric.

    Parameters
    ----------
    labels : List[int]
        List of notes labels (0 or 1).
    scores : List[float]
        List of notes scores.
    k : int, optional
        Number of notes to consider in the metric.

    Returns
    -------
    float
        The Recall@k metric value.
    """
    tp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),
                               key=lambda x: x[1])[::-1][:k] if x[0] == 1].count(1)
    fn = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),
                               key=lambda x: x[1])[::-1][k:] if x[0] == 1].count(1)
    recall_k = tp / (tp + fn)

    return recall_k


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """
    Computes the Precision@k metric.

    Parameters
    ----------
    labels : List[int]
        List of notes relevance labels (0 or 1).
    scores : List[float]
        List of notes scores.
    k : int, optional
        Number of notes to consider in the metric.

    Returns
    -------
    float
        The Precision@k metric value.
    """
    tp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  # Это вроде верно
                               key=lambda x: x[1])[::-1][:k] if x[0] == 1].count(1)
    fp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),
                               key=lambda x: x[1])[::-1][:k] if x[0] == 0].count(0)
    precision_k = tp / (tp + fp)

    return precision_k

def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Notes relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    iDCG_lst = sorted(relevance)[::-1]
    iDCG = 0
    DCG = 0
    if method == 'standard':
      iDCG = sum([iDCG_lst[i]/math.log2(i+2) for i in range(k)])
      DCG = sum([relevance[i]/math.log2(i+2) for i in range(k)])
    elif method == 'industry':
      iDCG = sum([(2**iDCG_lst[i]-1)/math.log2(i+2) for i in range(k)])
      DCG = sum([(2**relevance[i] - 1)/math.log2(i+2) for i in range(k)])

    try:
        score = DCG/iDCG
        return score
    except ZeroDivisionError:
        pass

def calculate_metrics_for_notes(path, k):
    """
    Calculates precision, recall, and nDCG metrics for a given dataset of notes.

    Parameters
    ----------
    path : str
        Path to the dataset file.
    k : int
        Number of notes to consider in the metrics.

    Returns
    -------
    float
        Average precision value.
    float
        Average recall value.
    float
        Average nDCG value.
    """

    data = pd.read_csv(path)

    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    num_notes = len(data)

    for actual, predicted in zip(data['linked_notes_bin'], data['linked_note_model']):

        precision = precision_at_k(actual, predicted, k)
        recall = recall_at_k(actual, predicted, k)
        ndcg = normalized_dcg(actual, predicted, k)

        precision_sum += precision
        recall_sum += recall
        ndcg_sum += ndcg

    precision_avg = precision_sum / num_notes
    recall_avg = recall_sum / num_notes
    ndcg_avg = ndcg_sum / num_notes

    return "Precision: " + str(precision_avg), "Recall: " + str(recall_avg), "nDCG: " + str(ndcg_avg)