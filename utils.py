import numpy as np
from prob_accum import ProbabilityAccumulator
from scipy.stats.mstats import mquantiles
from coverage import wsc_unbiased


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

def calibrate_full(i, probs, labels, alpha):
    probs = np.concatenate([probs[:i], probs[i+1:]])
    labels = np.concatenate([labels[:i], labels[i+1:]])
    return calibrate(probs, labels, alpha)

def evaluate_predictions(S, X, y, display=False):
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # wsc_coverage = wsc_unbiased(X, y, S)
    length = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    length_cover = np.mean([len(S[i]) for i in idx_cover])
    if display:
        print('Marginal coverage:       {:2.3%}'.format(marg_coverage))
        # print('WS conditional coverage: {:2.3%}'.format(wsc_coverage))
        print('Average size:            {:2.3f}'.format(length))
        print('Average size | coverage: {:2.3f}'.format(length_cover))
    return marg_coverage, length, length_cover
