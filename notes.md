# Stage 1: autoencoder (no deps, can run all datasets in parallel)
python -m pipeline.cli autoencoder --dataset hcrl_ch --model-size teacher
python -m pipeline.cli autoencoder --dataset hcrl_sa --model-size teacher
python -m pipeline.cli autoencoder --dataset set_01 --model-size teacher
python -m pipeline.cli autoencoder --dataset set_02 --model-size teacher
python -m pipeline.cli autoencoder --dataset set_03 --model-size teacher
python -m pipeline.cli autoencoder --dataset set_04 --model-size teacher

# Stage 2: curriculum (needs autoencoder done)
python -m pipeline.cli curriculum --dataset <ds> --model-size teacher

# Stage 3: fusion (needs autoencoder + curriculum done)
python -m pipeline.cli fusion --dataset <ds> --model-size teacher


python -m pipeline.cli autoencoder --dataset <ds> --model-size student
python -m pipeline.cli curriculum  --dataset <ds> --model-size student
python -m pipeline.cli fusion      --dataset <ds> --model-size student


pip install "pulp<3"
snakemake -s pipeline/Snakefile -n    # dry run to verify
snakemake -s pipeline/Snakefile --profile profiles/slurm --jobs 20   # actual run
