=== Run information ===

Scheme:       weka.classifiers.functions.Logistic -R 1.0E-8 -M -1
Relation:     ?????4attr
Instances:    11042
Attributes:   5
              PPERSAUT
              PVRAAUT
              PBRAND
              PPLEZIER
              CARAVAN
Test mode:    user supplied test set:  size unknown (reading incrementally)

=== Classifier model (full training set) ===

Logistic Regression with ridge parameter of 1.0E-8
Coefficients...
                               Class
Variable                           0
====================================
PPERSAUT=0                    0.3472
PPERSAUT=1                         0
PPERSAUT=2                         0
PPERSAUT=3                         0
PPERSAUT=4                    89.218
PPERSAUT=5                    0.5817
PPERSAUT=6                   -1.0625
PPERSAUT=7                   33.0259
PPERSAUT=8                   40.7901
PPERSAUT=9                         0
PVRAAUT=0                  -174.5524
PVRAAUT=1                          0
PVRAAUT=2                          0
PVRAAUT=3                          0
PVRAAUT=4                   -71.2762
PVRAAUT=5                          0
PVRAAUT=6                   261.0045
PVRAAUT=7                          0
PVRAAUT=8                          0
PVRAAUT=9                  -184.9294
PBRAND=0                      0.2017
PBRAND=1                       1.093
PBRAND=2                      1.4591
PBRAND=3                     -0.3364
PBRAND=4                     -0.7263
PBRAND=5                      0.2295
PBRAND=6                       1.527
PBRAND=7                     16.4985
PBRAND=8                    152.6155
PBRAND=9                           0
PPLEZIER=0                   -0.5641
PPLEZIER=1                   -4.2929
PPLEZIER=2                   -2.3474
PPLEZIER=3                   -3.5046
PPLEZIER=4                   -2.8993
PPLEZIER=5                  439.9419
PPLEZIER=6                   -3.8195
PPLEZIER=7                         0
PPLEZIER=8                         0
PPLEZIER=9                         0
Intercept                   175.6629


Odds Ratios...
                               Class
Variable                           0
====================================
PPERSAUT=0                    1.4151
PPERSAUT=1                         1
PPERSAUT=2                         1
PPERSAUT=3                         1
PPERSAUT=4      5.583370065997361E38
PPERSAUT=5                    1.7891
PPERSAUT=6                    0.3456
PPERSAUT=7     2.2027320509675772E14
PPERSAUT=8     5.1872076689187904E17
PPERSAUT=9                         1
PVRAAUT=0                          0
PVRAAUT=1                          1
PVRAAUT=2                          1
PVRAAUT=3                          1
PVRAAUT=4                          0
PVRAAUT=5                          1
PVRAAUT=6      2.253374199138177E113
PVRAAUT=7                          1
PVRAAUT=8                          1
PVRAAUT=9                          0
PBRAND=0                      1.2235
PBRAND=1                      2.9831
PBRAND=2                       4.302
PBRAND=3                      0.7143
PBRAND=4                      0.4837
PBRAND=5                      1.2579
PBRAND=6                      4.6042
PBRAND=7               14628367.0069
PBRAND=8       1.9058389222236917E66
PBRAND=9                           1
PPLEZIER=0                    0.5689
PPLEZIER=1                    0.0137
PPLEZIER=2                    0.0956
PPLEZIER=3                    0.0301
PPLEZIER=4                    0.0551
PPLEZIER=5    1.1597187850522429E191
PPLEZIER=6                    0.0219
PPLEZIER=7                         1
PPLEZIER=8                         1
PPLEZIER=9                         1


Time taken to build model: 1.14 seconds

=== Evaluation on test set ===

Time taken to test model on supplied test set: 0.1 seconds

=== Summary ===

Correctly Classified Instances        2648               66.2    %
Incorrectly Classified Instances      1352               33.8    %
Kappa statistic                          0.0866
Mean absolute error                      0.4043
Root mean squared error                  0.4573
Relative absolute error                 80.2571 %
Root relative squared error             90.7846 %
Coverage of cases (0.95 level)          99.875  %
Mean rel. region size (0.95 level)      99.475  %
Total Number of Instances             4000     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.665    0.378    0.965      0.665    0.787      0.142    0.692     0.966     0
                 0.622    0.335    0.105      0.622    0.180      0.142    0.692     0.123     1
Weighted Avg.    0.662    0.376    0.914      0.662    0.751      0.142    0.692     0.916     

=== Confusion Matrix ===

    a    b   <-- classified as
 2500 1262 |    a = 0
   90  148 |    b = 1

