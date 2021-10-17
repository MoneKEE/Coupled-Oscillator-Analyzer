from scipy.optimize._lsq.common import solve_trust_region_2d
# from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss, accuracy_score, precision_score, mean_absolute_error, mean_squared_error,roc_auc_score, explained_variance_score
import datacapture as dc
from datetime import datetime as dt
from datetime import timedelta
import modes


class MLP(MLPClassifier):
  def __init__(self, data, pos,asset):
    super().__init__()
    self.enddt = dt.now()
    self.startdt = timedelta(days=-365)
    
    self.learning_rate = 'constant'
    self.random_state = 123
    self.activation = 'tanh'
    self.solver = 'adam'
    self.shuffle = False
    self.hidden_layer_sizes=(22,22,)
    self.warm_start=True
    self.fit(data,pos)