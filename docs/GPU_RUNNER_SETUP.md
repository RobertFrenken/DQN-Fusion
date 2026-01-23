# GPU Runner Setup (Self-hosted GitHub Actions Runner)

This guide helps you provision a self-hosted GitHub Actions runner equipped with an NVIDIA GPU for running the GPU smoke-suite.

Prerequisites
-------------
- A Linux machine with an NVIDIA GPU and sufficient CPU/RAM/disk.
- Administrator/root access on the machine.
- GitHub repository admin access to create a runner registration token.

Quick steps
-----------
1. On GitHub, go to: **Settings → Actions → Runners → New self-hosted runner**, pick the OS and follow instructions to generate a temporary token.
2. On the runner host, copy this repository. Place the generated token in the environment variable `RUNNER_TOKEN` and set `GITHUB_REPO` to your `<owner>/<repo>`.
3. Run the helper script (from repository root):

   ```bash
   GITHUB_REPO=your-org/your-repo \
   RUNNER_TOKEN=<the-token> \
   ./scripts/setup_selfhosted_runner.sh
   ```

4. Confirm the runner appears under your repo's Actions → Runners and has the label `gpu`.

Systemd & service
-----------------
The `setup_selfhosted_runner.sh` helper uses the Actions runner's provided `svc.sh` script to install a service that starts on boot. You can manage it with:

```bash
cd /opt/github-runner
./svc.sh stop
./svc.sh start
./svc.sh uninstall
```

Security and operational notes
------------------------------
- Treat the runner as a privileged machine — it executes code from the repository.
- Limit network egress and SSH access to trusted admins.
- Rotate the registration token if the host is decommissioned or compromised.
- Keep system packages and NVIDIA drivers up to date.

Testing the runner
------------------
Once the runner is online and labeled (`gpu`), you can trigger the GPU smoke workflow manually from the repository Actions tab (`GPU Smoke Test`) or via the `workflow_dispatch` event. The workflow will run on a runner that matches `self-hosted,gpu` labels.

Troubleshooting
---------------
- If `nvidia-smi` fails, ensure NVIDIA drivers and CUDA are installed.
- Ensure Python & dependencies are available; the workflow installs `requirements.txt` at runtime.
- Check runner logs in `/opt/github-runner/_diag` for diagnostic details.

Maintenance
-----------
- Schedule regular updates and security patches for the host.
- Monitor disk usage and rotate or prune old artifacts from `experimentruns_test/` or `experiment_runs/`.

Contact
-------
If you'd like, I can help prepare a minimal AMI or image that includes CUDA + drivers + the runner preconfigured (requires cloud provider access).