import numpy as np 
import matplotlib.pyplot as plt

all_caps = ["iou", "gt"]
def clean(s):
    s = s.split('_')
    return " ".join([ss.upper() if ss in all_caps else ss.capitalize() for ss in s])

def uncertainty():
    H = np.load('uncertainty_matrix.npy')
    # H = H.transpose()
    x,y = np.nonzero(H)

    ticks=[1+.0005*i for i in range(6)]
    print(ticks)
    plt.scatter(x,y,s=2,c='k')
    plt.yticks([0,5,10,15,20,25], ticks)
    plt.ylabel('Uncertainty')
    plt.xlabel('Pixel Darkness')
    plt.show()

met = [
    "precision", "recall", "iou",
    "seedless_precision","seedless_recall","seedless_iou",
    "count_guesses","count_gt","count_intersection","count_union",
    "count_seedless_guesses","count_seedless_gt","count_seedless_intersection","count_seedless_union"
]
met_idx = {m:i for i,m in enumerate(met)}
H = np.load('all_metrics.npy')
met_lists = {m:H[:,met_idx[m]].reshape(-1) for m in met}

def hist(metric, bins=20):
    assert (metric is not None)
    vals = met_lists[metric]
    plt.hist(vals, bins=bins, log=True, color='k')
    plt.xlabel(clean(metric))
    plt.ylabel("log(Count)")
    plt.show()

def scatter(metric1, metric2, logx=True, logy=False):
    assert (metric1 is not None)
    assert (metric2 is not None)
    x = met_lists[metric1]
    y = met_lists[metric2]
    plt.scatter(x,y,s=1,c='k')
    plt.xlabel(clean(metric1))
    plt.ylabel(clean(metric2))
    if logx:
        plt.xscale('symlog')
    if logy:
        plt.yscale('symlog')
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pick metrics")
    parser.add_argument('--metric1')
    parser.add_argument('--metric2')
    parser.add_argument('--logx', action='store_true')
    parser.add_argument('--logy', action='store_true')
    parser.add_argument('--bins', type=int, default=20)
    args = parser.parse_args()

    if args.metric1 and args.metric2:
        scatter(args.metric1, args.metric2, args.logx, args.logy)
    elif args.metric1:
        hist(args.metric1, args.bins)
    else:
        raise ValueError
