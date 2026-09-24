# Примеры предсказаний и ошибок

Порог: 0.7

Примеры отобраны автоматически по пересечению масок и числу ошибок.

## Удачные

- [test/20416.png](good_1.png): IoU=0.9409615837850578, TP=15970, FP=682, FN=320
- [test/20378.png](good_2.png): IoU=0.9382126756828427, TP=3538, FP=125, FN=108
- [test/20834.png](good_3.png): IoU=0.919185591229444, TP=5869, FP=224, FN=292

## Плохо выделенные

- [test/20625.png](bad_1.png): IoU=0.0060020876826722335, TP=23, FP=236, FN=3573
- [test/20640.png](bad_2.png): IoU=0.03730797366495977, TP=51, FP=0, FN=1316
- [test/20772.png](bad_3.png): IoU=0.03750794659885569, TP=177, FP=2, FN=4540

## Пропущенные дефекты

- [test/20969.png](missed_1.png): IoU=0.0, TP=0, FP=0, FN=3952
- [test/20141.png](missed_2.png): IoU=0.0, TP=0, FP=0, FN=3714
- [test/20744.png](missed_3.png): IoU=0.0, TP=0, FP=0, FN=3185

## Ложные срабатывания

- [test/20381.png](false_positive_1.png): IoU=0.0, TP=0, FP=26263, FN=0
- [test/20267.png](false_positive_2.png): IoU=0.0, TP=0, FP=16466, FN=0
- [test/20057.png](false_positive_3.png): IoU=0.0, TP=0, FP=6764, FN=0
