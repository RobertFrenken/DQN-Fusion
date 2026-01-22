I had to delete all of the old runs for a couple of reasons:
- First is that I was getting loading errors with the pickle files
- Second the path creation and saving is all messed up, fusion was under gat folders for some reason and this was causing the dqn models to freak out and the dqn models were trying to backup find paths that I deleted as they were old. What needs to happen is to seriously restructure the way models are saved and be strict on the appropiate paths for a given run. No fallbacks, if it isn't there give an error when you save a model it needs to be in a consisent place and it should be printed out. 
On Slurm outputs:
- The slurm job output needs to have contextual knowledge on the success of the job. Saying job completion without checking for errors doesn't help.

On file paths:
- Every single python file needs to be compatible with pytorch lightning and its utility functions. No old dependencies, no old code, no fallbacks, it either works or it crashes with an informative error.
On refactoring issues:
- The trainer python file that train_with_hydra_zen is now broken up in different parts into src.training. This has been causing issues and needs to be updated to the new dependency format
On new folder path grammer:
- in osc_job_manager.py I have started creating a new configuration setup to properly document each possible combination of model run.
- There needs to be a new pathing system to document each experiment which will take the following levels:
level 1: experiment_runs - This is the parent path that all models will be documented in
level 2: modality- lght now all automotive, but will expand to other types like physical systems and internet datasets
level 3: dataset (hcrl_ch, set_01, set_02, set_03, set_04): The specific dataset within that modality
level 4: learning type (unsupervised, classifier, fusion): The type of learning being conducted
level 5: model architecture (VGAE, GAT, DQN): this should have functionality for potential variations like generative GANs, LSTMS, other fusion mechancis
level 6: model size (Teacher, Student): this should have functionality for potential variations like an intermediate, a huge, or a tiny size 
level 7: Distillation (yes, no): This is above training type as a configuration could have both, which is why it is on a higher path level
level 8: training type (all samples, normal only , curriculum schedule classifier, curriculum schedule fusion, etc): This is the specific training strategy
level 9: Here the saved model should sit here, along with it's training metrics in a folder, the validation metrics, and when it is tested on the test set the evaluation results will be put into its own folder. Right now in the datasets it is split between train_ files and test_ files, with the test_ files in the set_xx datasets having unique tests for known/ unknown attacks and known/ unknown vehicles. I will need guidance here on the best way to orgainize these particular evaluations.
On model naming and saving:
- The models should be saved as a dictionary of the model weights in a file type that will not run into issues as the pickle files have
- the models need a descriptive name so that I can easily trace the path down and find the saved model. This was a big issue earlier

On MLFlow:
- I want MLflow to save the training metrics with that particular saved model, and I want the configuration of the GUI to be more comprehensive so when I launch an instance the UI will display strong organization of each type

On train_with_hyrda_zen:
- It looks like from around lines 192-272 there is a chunk of code that is present but dulled out by the linter implying that it will never be run. I want to make sure this is no longer needed, and if not I want to remove that section.



Hydra-Zen Configuration System for CAN-Graph

This module replaces the YAML configuration files with type-safe Python configurations
using hydra-zen. Benefits:
- Type safety and IDE support
- Programmatic config generation
- Reduced file clutter
- Better validation
- Dynamic configuration composition

Looking to change the configuration to the new pathing system:

        self.osc_settings = {
            "account": "PAS3209",  # Your account
            "email": "frenken.2@osu.edu",  # Your email (used for SBATCH notifications if enabled)
            "project_path": str(self.project_root),  # Project path (keeps previous behaviour but derived)
            "conda_env": "gnn-gpu",  # Your conda environment
            "notify_webhook": "",  # Optional: Slack/Teams webhook URL for concise completion notifications
            "notify_email": "frenken.2@osu.edu"     # Optional: single email to receive job completion summaries (one per job)
        }

        self.osc_parameters = {
            "wall_time": "02:00:00",
            "memory": "32G",
            "cpus": 8,
            "gpus": 1
        }

        self.training_configurations = {
            "modalities": ["automotive", "internet", "water_treatment"],
            # will need to handle the pathing based on modalities in future
            # right now automotive is the only modality used
            "datasets": {"automotive": ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"],
                         "internet": [],
                         "water_treatment": []},
            "learning_types": ["unsupervised", "supervised", "rl_fusion"],
            "model_architectures": {"unsupervised": ["vgae"], "supervised": ["gat"], "rl_fusion": ["dqn"]},
            # right now small = student and teacher = large with options to expand
            "model_sizes": ["small", "medium", "large"],
            "distillation": ["no", "standard"],
            "training_modes": ["all_samples", "normals_only","curriculum_classifier", "curriculum_fusion"],
        }