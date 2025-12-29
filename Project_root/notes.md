File Structure:
CAN-Graph/
│
├── models/
│   ├── models.py - holds the classifier, AD, and in future fusion methods
│   └── pipeline.py - holds the pipeline class to train the models
│   └── __init__.py
├── preprocessing/
│   ├── preprocessing.py - will take in the root folder name, and return the processed graphs
├── training/
│   ├── AD_KD_GPU.py - Takes in the teacher models of both GAT and AD, will train the smaller student models
│   └── AD-KD.py
|   └── osc_training_AD.py - trains the teacher models of both GAT and AD
|   └── training_utils.py - has the DistillationTrainer utility class
|   └── training-AD.py
├── evaluation/
│   └── evaluation.py
├── utils.py
├── conf/
│   └── base.yaml
├── tests/
│   └── test_models.py
└── README.md