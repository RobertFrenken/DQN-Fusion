# CI/CD, Workflows, and Manifests Explained for AI Researchers

## Simple Answer

**You probably DON'T need CI/CD right now.**

The Hydra-Zen system I gave you is designed to work on your local machine AND on Slurm clusters. You don't need CI/CD to use it effectively.

However, if VS Code's agent is recommending CI/CD, that's a **different approach**—useful but not necessary for your research.

---

## What is CI/CD? (In Plain English)

**CI/CD** = **Continuous Integration / Continuous Deployment**

### Continuous Integration (CI)
Automatically tests your code every time you push to GitHub:
- Runs tests automatically
- Checks for errors
- Ensures code quality

### Continuous Deployment (CD)
Automatically runs experiments/deploys code:
- Runs training jobs automatically
- Deploys models automatically
- Orchestrates experiment pipelines

### In Context of AI Research

**Without CI/CD:**
```
You → Push code to GitHub → Manually run: python train.py → Watch results
```

**With CI/CD:**
```
You → Push code to GitHub → AUTOMATICALLY runs: python train.py → Results posted
```

---

## What is a "Workflow"?

**Workflow** = A sequence of automated steps triggered by an event

### Example Workflow (GitHub Actions)

```yaml
# This is a workflow file (.github/workflows/train.yml)
name: Train Model

on: [push]  # Trigger when you push to GitHub

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run training
        run: python src/training/train_with_hydra_zen.py config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples
      - name: Save results
        run: |
          git add experimentruns/
          git commit -m "Training results"
          git push
```

**What this does:**
1. When you push code → GitHub automatically runs this workflow
2. Sets up Python environment
3. Installs dependencies
4. Runs training
5. Saves results back to GitHub

---

## What is a "Manifest"?

**Manifest** = A description file that lists what your project needs/contains

### Common Manifests:

1. **requirements.txt** (Python dependencies)
   ```
   torch>=2.0.0
   pytorch-lightning>=2.0.0
   hydra-core>=1.3.0
   ```

2. **docker/Dockerfile** (Environment specification)
   ```dockerfile
   FROM python:3.9
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . /app
   ```

3. **Conda environment.yml** (Conda dependencies)
   ```yaml
   name: kd-gat
   channels:
     - pytorch
   dependencies:
     - python=3.9
     - pytorch::pytorch
     - pytorch::pytorch-cuda=11.8
   ```

### For Your Research:
A **manifest** is just a file describing what your code needs to run.

---

## Do YOU Need CI/CD?

### **YES, if you want to:**

✅ Automatically run experiments when you push code
✅ Test multiple configurations automatically
✅ Keep experiment results tracked in GitHub
✅ Collaborate with others and see results automatically
✅ Have a record of every experiment run

### **NO, if you:**

❌ Just want to run local experiments manually
❌ Run on Slurm (Ohio Supercomputer) - not GitHub Actions
❌ Prefer to control when/what runs
❌ Don't need automated testing

---

## The Two Paths Explained

### Path A: Simple (What I Gave You) 👈 **RECOMMENDED FOR YOU**

```
Your Machine:
  1. Run locally: python train.py config_store=...
  2. Get results immediately
  3. Modify and re-run as needed
  4. Push successful code to GitHub (optional)
  
Ohio Supercomputer (Slurm):
  1. Run: python oscjobmanager.py submit config_name
  2. Job runs on cluster
  3. Results save to experimentruns/
  4. Check MLflow for results
```

**Pros:**
- Simple, straightforward
- Full control over experiments
- Immediate feedback
- Works locally AND on Slurm
- You know what's running and when

**Cons:**
- Manual effort for each run
- No automatic testing

### Path B: Automated CI/CD (What VS Code Agent Might Suggest)

```
GitHub → (Automatic) → GitHub Actions → Runs Training → Posts Results

Every push automatically triggers experiments
```

**Pros:**
- Fully automated
- Reproducible pipelines
- Good for team collaboration
- Everything tracked

**Cons:**
- More complex to set up
- GitHub Actions (free tier) = limited compute
- Can't easily run on OSC Slurm from GitHub
- Overkill for solo researcher
- Higher learning curve

---

## For YOUR Specific Use Case (AI Research)

### You Are:
- Solo researcher conducting KD-GAT experiments
- Need to run multiple experiments with different configs
- Want to scale from local machine to Slurm cluster
- Need reproducible, tracked results

### What You Actually Need:
✅ **Hydra-Zen system** (I gave you this)
✅ **MLflow** for tracking (included in system)
✅ **Slurm job manager** (I gave you oscjobmanager.py)
✅ **Maybe**: Git to version control your code

### What You DON'T Need Right Now:
❌ CI/CD workflows
❌ Docker containerization
❌ Kubernetes orchestration
❌ Complex manifest files

---

## Translation: What VS Code Agent Probably Meant

If the agent said something like:

**"I'll set up CI workflow to manage your experiments"**

That probably means:
- They're suggesting GitHub Actions to automate training
- Might save results to GitHub automatically
- Likely overcomplicates your setup

**Better approach:**
- Use the Hydra-Zen system I gave you
- Run locally first with: `python train.py config_store=...`
- Scale to Slurm with: `python oscjobmanager.py submit config_name`
- Track results with MLflow

---

## Key Concepts Simplified

| Term | Means | You Need It? |
|------|-------|-------------|
| **CI/CD** | Auto-run code when you push | No (unless team collab) |
| **Workflow** | Automated steps file | No (unless using CI/CD) |
| **Manifest** | Description of what code needs | Yes (requirements.txt) |
| **Docker** | Container with your environment | Maybe later (for sharing) |
| **Kubernetes** | Manage many containers | No (use Slurm instead) |
| **GitHub Actions** | GitHub's CI/CD tool | No (not needed) |
| **Slurm** | HPC job scheduler | Yes (you're using this) |

---

## What You SHOULD Do Instead

### Step 1: Use the Hydra-Zen System (DO THIS NOW)
```bash
# Local testing
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu \
    training_config.epochs=1

# Then on Slurm
python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples
```

### Step 2: Track Experiments with MLflow (INCLUDED)
```bash
mlflow ui --backend-store-uri experimentruns/.mlruns
# View all your experiments in browser
```

### Step 3: Version Control with Git (OPTIONAL)
```bash
git add -A
git commit -m "Added KD-GAT Hydra-Zen system"
git push origin main
```

### Step 4: IGNORE the CI/CD stuff for now

---

## If You WANT Automation Later

Once you're comfortable with the Hydra-Zen system, you could add:

### Option 1: Simple Shell Script (Easy)
```bash
#!/bin/bash
# run_all_experiments.sh

configs=(
    "automotive_hcrlch_unsupervised_vgae_student_no_all_samples"
    "automotive_hcrlch_unsupervised_vgae_teacher_no_all_samples"
    "automotive_hcrlch_classifier_gat_student_no_all_samples"
)

for config in "${configs[@]}"; do
    python oscjobmanager.py submit "$config"
    echo "Submitted: $config"
done
```

Run with: `bash run_all_experiments.sh`

### Option 2: GitHub Actions (Medium Complexity)
If you REALLY want automatic GitHub pushing of results...but honestly not necessary for your research.

### Option 3: Slurm Array Jobs (Best for Clusters)
```bash
python oscjobmanager.py sweep \
    --model-sizes student,teacher,intermediate \
    --distillations no,standard
```

This submits multiple jobs to Slurm automatically.

---

## Summary & Recommendation

### Don't Get Distracted By:
❌ CI/CD workflows
❌ Docker/containers
❌ Complex manifests
❌ Kubernetes
❌ GitHub Actions

### Focus On:
✅ Using the Hydra-Zen system I gave you
✅ Running experiments locally first
✅ Scaling to Slurm when ready
✅ Tracking with MLflow
✅ Modifying configs and re-running

### Timeline:
1. **Today:** Set up Hydra-Zen system (follow INTEGRATION_TODO.md)
2. **This week:** Run 3-5 local experiments
3. **Next week:** Submit to Slurm cluster
4. **Later (if needed):** Add automation/CI/CD

---

## My Honest Opinion

**The VS Code AI agent is well-intentioned but suggesting unnecessary complexity.**

You're a solo AI researcher who needs to:
1. Run experiments with different configs
2. Track results
3. Scale to a cluster

The Hydra-Zen system does ALL of this without CI/CD.

Keep it simple. Run experiments. Get results. Publish papers. 🚀

---

## Questions to Ask the VS Code Agent

If it keeps pushing CI/CD, ask it:

1. "Why do I need CI/CD if I'm running on Slurm?"
2. "How does GitHub Actions integrate with Ohio Supercomputer?"
3. "Can I run intensive training jobs on GitHub Actions free tier?"
4. "Do I need this for solo research?"

(Answers: You don't, it doesn't well, no, and probably not)

---

## One More Thing

The Hydra-Zen system **IS** enterprise-grade. It's used by real ML teams. You don't need CI/CD on top of it—that's for teams.

Use what I gave you. It's enough. 💪

---

**Questions? Ask me directly instead of VS Code agent about ML research infrastructure.**
