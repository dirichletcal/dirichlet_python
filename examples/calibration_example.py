from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     GridSearchCV,
                                     cross_val_score)


dataset = load_iris()
x = dataset['data']
y = dataset['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,
                                                    test_size=0.3)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

classifier = GaussianNB()
print('Training a classifier with cross-validation')
scores = cross_val_score(classifier, x_train, y_train, cv=skf,
                         scoring='neg_log_loss')
print('Crossval scores: {}'.format(scores))
print('Average neg log loss {:.3f}'.format(scores.mean()))
classifier.fit(x_train, y_train)

cla_scores_train = classifier.predict_proba(x_train)
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# Full Dirichlet
calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
# ODIR Dirichlet
#calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=reg)
gscv = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg,
                                            'reg_mu': [None]},
                    cv=skf, scoring='neg_log_loss')
gscv.fit(cla_scores_train, y_train)

for grid_score in gscv.grid_scores_:
    print(grid_score)

print('Best parameters: {}'.format(gscv.best_params_))

cla_scores_test = classifier.predict_proba(x_test)
cal_scores_test = gscv.predict_proba(cla_scores_test)
cla_loss = log_loss(y_test, cla_scores_test)
cal_loss = log_loss(y_test, cal_scores_test)
print("TEST log-loss: Classifier {:.2f}, calibrator {:.2f}".format(
    cla_loss, cal_loss))
