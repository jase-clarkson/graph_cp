import numpy as np
import pandas as pd
from prob_accum import ProbabilityAccumulator
from scipy.stats.mstats import mquantiles
from scipy.optimize import brentq
from coverage import wsc_unbiased

def get_weighted_quantile(scores, weights, alpha):
    wtildes = weights / (weights.sum() + 1)
    def critical_point_quantile(q): return (wtildes * (scores <= q)).sum() - (1 - alpha)
    q = brentq(critical_point_quantile, -100, 100)
    return q


def calibrate_weighted(probs, labels, weights, alpha):
    n = probs.shape[0]
    if n == 0:
        return alpha
    # Calibrate
    calibrator = ProbabilityAccumulator(probs)
    eps = np.random.uniform(low=0, high=1, size=n)
    alpha_max = calibrator.calibrate_scores(labels, eps)
    scores = alpha - alpha_max
    alpha_correction = get_weighted_quantile(scores, weights, alpha)
    return alpha - alpha_correction


def calibrate(probs, labels, alpha):
    n = probs.shape[0]
    if n == 0:
        return alpha
    # Calibrate
    calibrator = ProbabilityAccumulator(probs)
    eps = np.random.uniform(low=0, high=1, size=n)
    alpha_max = calibrator.calibrate_scores(labels, eps)
    scores = alpha - alpha_max
    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n))
    alpha_correction = mquantiles(scores, prob=level_adjusted)
    return alpha - alpha_correction



def predict(probs, alpha, allow_empty=True):
    n = probs.shape[0]
    eps = np.random.uniform(0, 1, n)
    predictor = ProbabilityAccumulator(probs)
    S_hat = predictor.predict_sets(alpha, eps, allow_empty)
    return S_hat


def get_ssvc(sets, y, alpha, bins=[-1] + [i for i in range(1, 3)] + [9]):
    sets = pd.Series(sets, name='set')
    sets = pd.DataFrame(sets)
    sets['set_size'] = sets['set'].apply(len)
    sets['covers'] = [y[i] in sets.loc[i, 'set'] for i in range(len(y))]
    l_cc = []
    for _, subset in sets.groupby(pd.cut(sets['set_size'], bins)):
        l_cc.append(subset['covers'].sum() / len(subset))
    print(l_cc)
    return max(np.abs(np.array(l_cc) - (1 - alpha)))


def evaluate_predictions(S, X, y, alpha, display=False):
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    sscv = get_ssvc(S, y, alpha)
    length = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    length_cover = np.mean([len(S[i]) for i in idx_cover])
    if display:
        print('Marginal coverage:       {:2.3%}'.format(marg_coverage))
        # print('WS conditional coverage: {:2.3%}'.format(wsc_coverage))
        print('Average size:            {:2.3f}'.format(length))
        print('Average size | coverage: {:2.3f}'.format(length_cover))
    return marg_coverage, length, length_cover, sscv
