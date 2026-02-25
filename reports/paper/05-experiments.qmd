---
title: "Experiments"
---

## Experiments

This section presents the experimental setup, evaluation metrics, and insights into the datasets used in this study.

### Evaluation Metrics

The performance of the model is evaluated using accuracy and F1-Score.

### Datasets

Our proposed method has been evaluated on three publicly available automotive CAN intrusion detection datasets, each offering distinct characteristics and challenges for comprehensive IDS evaluation.

#### HCRL Car-Hacking

This dataset contains CAN traffic from a Hyundai YF Sonata with four attack types: DoS, fuzzing, RPM spoofing, and gear spoofing. All attacks were conducted on a real vehicle, with data logged via the OBD-II port. The dataset includes 988,872 attack-free samples and approximately 16.6 million total samples across all attack types [@Song2020carhacking].

#### HCRL Survival Analysis

Collected from three vehicles (Chevrolet Spark, Hyundai YF Sonata, Kia Soul), this dataset enables scenario-based evaluation with three attack types: flooding (DoS), fuzzing, and malfunction (spoofing). The dataset is structured with 627,264 training samples and four testing subsets designed to evaluate IDS performance across known/unknown vehicles and known/unknown attacks [@Han2018survival].

#### can-train-and-test

The largest dataset, containing CAN traffic from four vehicles across two manufacturers (GM and Subaru). It provides nine distinct attack scenarios including DoS, fuzzing, systematic, various spoofing attacks, standstill, and interval attacks. The dataset is organized into four vehicle sets (set_01 to set_04) with over 192 million total samples. This dataset exhibits extreme class imbalance with attack-free to attack sample ratios ranging from 36:1 to 927:1 across different subsets. Each set contains one training subset and four testing subsets following the known/unknown vehicle and attack paradigm [@Lampe2024cantrainandtest]. This work limits evaluation to the known vehicle and attack testing set.
