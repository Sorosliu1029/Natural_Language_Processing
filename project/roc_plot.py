import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def roc_plot(classifier, X, y, plot_title):
  """
  Run classifier with cross-validation and plot ROC curves
  """
  cv = StratifiedKFold(n_splits=5)

  mean_tpr = 0.0
  mean_fpr = np.linspace(0, 1, 100)

  colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
  lw = 2

  i = 0
  for (train, test), color in zip(cv.split(X, y), colors):
      probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
      # Compute ROC curve and area the curve
      fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
      mean_tpr += interp(mean_fpr, fpr, tpr)
      mean_tpr[0] = 0.0
      roc_auc = auc(fpr, tpr)
      plt.plot(fpr, tpr, lw=lw, color=color,
              label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

      i += 1
      print("%dth fold done" % i)

  plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
          label='Luck')

  mean_tpr /= cv.get_n_splits(X, y)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  plt.plot(mean_fpr, mean_tpr, color='r', linestyle='--',
          label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(plot_title)
  plt.legend(loc="lower right")
  plt.show()
  return mean_fpr, mean_tpr


def roc_plot_nocv(classifiers, X, y, plot_title):
  """
  Run classifierS and plot ROC curves
  """
  lw = 2

  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
  colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
  aucs = []
  print("start training and validating...")
  for classifier, color in zip(classifiers, colors):
    classifier_name = str(type(classifier)).split('.')[-1][:-2]
    probas_ = classifier.fit(train_X, train_y).predict_proba(test_X)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    aucs.append({classifier_name: roc_auc})
    plt.plot(fpr, tpr, lw=lw, color=color,
            label='%s ROC (area = %0.2f)' % (classifier_name, roc_auc))
    print("%s done" % classifier_name)

  plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
          label='Luck')

  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(plot_title)
  plt.legend(loc="lower right")
  plt.show()
  return aucs