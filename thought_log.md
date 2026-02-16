
2.16.26
- need to set up tmux configs so there can be multiple panes
and I can run claude within a session
- similar, update preferences to dotfiles repo across all projects
- mcp connection to my repos
- start testing langchain and langgraph
- investigate snakemake pipeline as it seems a little rule fragile to rerun
experiments if one is already tested. How can I run more if it must overwrite
the previous one?
- This also highlights the fragility in the exporting and visualization
if it only takes the latest models. Will need some sort of querying
or research on how to do this.
-snakemake had a failed jobs as they all raced to write to the sql database.
Need to add a retry method or investigate better methods
- claude starts in base python and doesn't correctly load in the miniconda
env. Need to add a bash setup file that I think my local projects have.
- Will need to continue enhancing the snakemake pipeline with additional features
- Also really would like a pipeline visual so I can understand the process