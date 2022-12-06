from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_prediction(y1, y2, N_total, n_toplot=10**10,):
    
    
    idxs = np.arange(len(y1))
    np.random.shuffle(idxs)
    
    y_expected = y1.reshape(-1)[idxs[:n_toplot]]
    y_predicted = y2.reshape(-1)[idxs[:n_toplot]]

    xy = np.vstack([y_expected, y_predicted])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_expected[idx], y_predicted[idx], z[idx]
    
    fig = plt.figure(figsize=(6,6))
    plt.title("Model Evaluation", fontsize=17)
    plt.ylabel('Modeled SMB (m.w.e)', fontsize=16)
    plt.xlabel('Reference SMB (m.w.e)', fontsize=16)
    sc = plt.scatter(y_plt, ann_plt, c=z, s=20)
    plt.clim(0.0,8.0)
    plt.tick_params(labelsize=14)
    plt.colorbar(sc) 
    lineStart = -1.5
    lineEnd = 1.5
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    plt.axvline(0.0, ls='-.', c='k')
    plt.axhline(0.0, ls='-.', c='k')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$R^2=%.2f$' % (r2_score(y_expected, y_predicted), ),
    r'$RMSE=%.2f$' % (mean_squared_error(y_expected, y_predicted), ),
    r'$N=%.0f$' % (N_total), ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #plt.show()

    return fig

def plot_scores(test_scores, train_scores):

    fig, (ax0) = plt.subplots(figsize=(6,3))
    ax0.plot(test_scores, label='test')
    ax0.plot(train_scores, label='train')
    ax0.set_ylabel('Score') #fontsize=25
    ax0.set_xlabel('n_estimators') #fontsize=25
    ax0.legend(loc='best')
    
    return fig


def plot_feature_importance(Reg_model, df_train_X, X_validation, y_validation):

    fig = plt.figure(figsize=(6,3))

    feature_importance = Reg_model.best_estimator_.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(df_train_X.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance( Reg_model, X_validation, y_validation,
    n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(df_train_X.columns)[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()

    return fig

