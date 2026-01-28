# SLURM Job Notifications Setup Guide

**Last Updated**: 2026-01-27

## Quick Setup

Add these two lines to your SLURM submission scripts (`.sh` files):

```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@osc.edu
```

## Mail Type Options

- `END` - Email when job completes successfully
- `FAIL` - Email when job fails
- `BEGIN` - Email when job starts
- `REQUEUE` - Email when job is requeued
- `ALL` - Email for all events (use sparingly!)

**Recommended**: `END,FAIL` (only notify on completion or failure)

## Example Script Header

```bash
#!/bin/bash
#SBATCH --job-name=vgae_training
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --account=PAS3209
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL          # <-- Add this
#SBATCH --mail-user=rf15@osc.edu      # <-- Add this (use your email)
```

## Updated Scripts

I've added notifications to:
- `test_run_counter_batch_size.sh`

**TODO**: Add to production training scripts when you create them.

## Email Format

OSC will send emails like:

```
Subject: SLURM Job_id=43978890 Name=test_run_counter_batch Ended, Run time 00:02:05

Body:
Job ID: 43978890
Cluster: owens
User/Group: rf15/PAS2022
State: COMPLETED
Cores: 16
CPU Utilized: 00:12:30
CPU Efficiency: 15.50%
Memory Utilized: 2.5 GB
Memory Efficiency: 7.81%
```

## Troubleshooting

### Not receiving emails?

1. **Check email address**: Verify it's your correct OSC email
2. **Check spam folder**: OSC emails may be filtered
3. **Verify account**: Run `sacctmgr show user $USER` to see your email on file
4. **Test with simple job**: Submit a quick 1-minute job to test

### Change your default email

```bash
# View current email
sacctmgr show user $USER

# Update email (contact OSC support if needed)
```

## Best Practices

1. **Use `END,FAIL` for long jobs** (>1 hour) - Know when they finish
2. **Skip notifications for quick tests** (<5 min) - Check manually
3. **Don't use `ALL`** - Too many emails, causes alert fatigue
4. **One email per job** - Keeps inbox manageable

## Integration with Monitoring

Combine with GPU monitoring:
1. Job starts → GPU monitor begins logging
2. Job ends → Email notification sent
3. You receive email → Run `python analyze_gpu_monitor.py gpu_monitor_JOBID.csv`
4. Check results and decide next steps

## Example Workflow

```bash
# 1. Submit long training job with notifications
sbatch train_vgae_full.sh

# 2. Go do other work, wait for email

# 3. Receive email "Job 12345 Ended"

# 4. Analyze results
python analyze_gpu_monitor.py gpu_monitor_12345.csv
tail -50 slurm-12345.out

# 5. If successful, submit next phase
sbatch train_gat_full.sh
```

---

**Status**: Notifications enabled for test scripts ✅
**Next**: Add to production training scripts when created
