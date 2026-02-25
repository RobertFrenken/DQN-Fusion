Small TODO log that the user finds through the codebase

- Orchestration tester job failed due to pytorch import error. I am assuming this is due to the script not loading in the correct uv package.
- There are still files or scripts using snakemake, these need to be cleaned up
- Really need to evaluate if the data pipeline is as lean and simple as it can be given our research
ambitions. The complexity should be from research, the data pipeline should be a means of assisting
us.
- I want to evaluate if all the output artifacts like slurm logs and wandb can be periodically cleaned
up? If we are tracking in our nice database, I would prefer these intermediate logs are more
ephemiercal (no longer than a week).
