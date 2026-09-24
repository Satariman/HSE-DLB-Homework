# ASVspoof 2019 — Synthetic Speech Detection

Проект решает задачу **детекции искусственно сгенерированной речи** на датасете **ASVspoof 2019 Logical Access**. Модель выполняет бинарную классификацию аудиозаписей на настоящую речь и синтезированную или преобразованную речь.

## Ключевые идеи решения

За основу решения была взята архитектура Light CNN (LCNN) с Max-Feature-Map (MFM), ранее применявшаяся в работах по ASVspoof 2019.

Аудиозаписи преобразуются в LFCC-признаки, которые затем подаются в LCNN. Выбор LFCC основан на сравнительных исследованиях frontend-представлений для этой задачи, где они показывали хорошие результаты совместно с LCNN.

Для классификации используется Cross Entropy. В loss используются веса классов для компенсации дисбаланса кол-ва классов в датасете.

## Использованные работы

Решение основано на следующих статьях:

1. **ASVspoof 2019: Automatic Speaker Verification Spoofing and Countermeasures Challenge Evaluation Plan** — описание задачи и датасета.
2. **A Light CNN for Deep Face Representation with Noisy Labels** — архитектура Light CNN и Max-Feature-Map.
3. **STC Antispoofing Systems for the ASVspoof2019 Challenge** — применение LCNN для anti-spoofing.
4. **A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection** — выбор frontend, loss-функции и параметров обучения.

## Результаты

Основная метрика — **EER (Equal Error Rate)**

| Metric | Result |
|---|---:|
| Best EER | **4.21%** |
| Epoch | **64** |

Оценка курс 10
При требовании EER < 5.3