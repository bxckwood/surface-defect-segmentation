# Примеры новой ResNet18 U-Net на test

Checkpoint: `runs/resnet18_unet_balanced_low_lr/best.pt` (эпоха 10). Порог 0.7 выбран на `val` до этой оценки. Показатели `test` здесь описательные: изображения набора уже просматривались при анализе исходной модели.

TP, FP, FN и IoU рассчитаны на сетке модели 640×256 (высота×ширина). Для панелей предсказанная маска возвращена к исходному размеру через NEAREST, разметка показана в исходном размере.

Панели созданы на основе [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) (Jakob Božič, Domen Tabernik, Danijel Skočaj; исходные изображения предоставила и разметила Kolektor Group d.o.o.). Они показывают изображение, разметку, предсказание и наложение. Для панелей действует [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Примеры отобраны автоматически по пересечению масок и числу ошибок.

## Удачные

- [test/20416.png](good_1.png): IoU=0.9347964560773641, TP=16143, FP=979, FN=147
- [test/20378.png](good_2.png): IoU=0.9324359989443125, TP=3533, FP=143, FN=113
- [test/20834.png](good_3.png): IoU=0.9277470355731225, TP=5868, FP=164, FN=293

## Плохо выделенные

- [test/20667.png](bad_1.png): IoU=0.049800796812749, TP=50, FP=0, FN=954
- [test/20193.png](bad_2.png): IoU=0.17009829699445816, TP=8410, FP=0, FN=41032
- [test/20801.png](bad_3.png): IoU=0.18157695223654283, TP=479, FP=2153, FN=6

## Пропущенные дефекты

- [test/20772.png](missed_1.png): IoU=0.0, TP=0, FP=0, FN=4717
- [test/20969.png](missed_2.png): IoU=0.0, TP=0, FP=0, FN=3952
- [test/20141.png](missed_3.png): IoU=0.0, TP=0, FP=0, FN=3714

## Ложные срабатывания

- [test/20358.png](false_positive_1.png): IoU=0.0, TP=0, FP=2536, FN=0
- [test/20894.png](false_positive_2.png): IoU=0.0, TP=0, FP=1293, FN=0
- [test/20304.png](false_positive_3.png): IoU=0.0, TP=0, FP=1173, FN=0
