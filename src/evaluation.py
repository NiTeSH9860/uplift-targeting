import numpy as np

# NumPy 2.x renamed trapz to trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _sort_by_score(score, t, y):
    """Order customers from highest predicted uplift to lowest."""
    order = np.argsort(-np.asarray(score), kind="stable")
    return np.asarray(t)[order], np.asarray(y)[order]


def qini_curve(score, t, y):
    """Cumulative incremental conversions as we contact more customers."""
    t, y = _sort_by_score(score, t, y)
    n = len(t)

    cum_treated = np.cumsum(t)
    cum_control = np.cumsum(1 - t)
    conv_treated = np.cumsum(y * t)
    conv_control = np.cumsum(y * (1 - t))

    # Scale control conversions to treated group size, then subtract
    ratio = cum_treated / np.maximum(cum_control, 1)
    qini = conv_treated - conv_control * ratio

    x = np.arange(1, n + 1) / n          # fraction of customers targeted
    return np.insert(x, 0, 0), np.insert(qini, 0, 0)


def auuc(score, t, y):
    """Area between the Qini curve and the random-targeting line."""
    x, q = qini_curve(score, t, y)
    random_line = x * q[-1]               # straight line from 0 to the final value
    return _trapz(q - random_line, x)


def uplift_at_k(score, t, y, k=0.3):
    """Treated-minus-control conversion rate among the top k fraction."""
    t, y = _sort_by_score(score, t, y)
    m = int(len(t) * k)
    t, y = t[:m], y[:m]
    if t.sum() == 0 or (1 - t).sum() == 0:
        return np.nan
    return y[t == 1].mean() - y[t == 0].mean()