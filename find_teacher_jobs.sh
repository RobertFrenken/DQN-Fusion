#!/bin/bash
# Helper script to find teacher DQN job IDs for dependency
# If multiple runs exist with same configuration, this script returns the LATEST job ID

echo "Finding teacher DQN job IDs..."
echo "================================"

# Use squeue to find currently queued/running DQN jobs
# Filter for teacher jobs (those WITHOUT "_s_" in the name which indicates student)
# Format: JobID|JobName|State
squeue -u $USER -o "%.18i|%.100j|%.8T" 2>/dev/null | \
  grep "dqn_" | \
  grep -v "dqn_s_" | \
  grep "fusion" | \
  sort -t'|' -k2 | \
  awk -F'|' '{
    name = $2
    jobid = $1
    state = $3

    # Trim whitespace
    gsub(/^[ \t]+|[ \t]+$/, "", name)
    gsub(/^[ \t]+|[ \t]+$/, "", jobid)
    gsub(/^[ \t]+|[ \t]+$/, "", state)

    # Extract dataset from job name
    if (match(name, /hcrl_sa/)) dataset = "hcrl_sa"
    else if (match(name, /hcrl_ch/)) dataset = "hcrl_ch"
    else if (match(name, /set_01/)) dataset = "set_01"
    else if (match(name, /set_02/)) dataset = "set_02"
    else if (match(name, /set_03/)) dataset = "set_03"
    else if (match(name, /set_04/)) dataset = "set_04"
    else dataset = "unknown"

    # Store latest job for each dataset
    if (dataset != "unknown" && !seen[dataset]++) {
      jobs[dataset] = jobid
      states[dataset] = state
      print dataset": "jobid" ("state")"
    }
  }
  END {
    # Print in order
    datasets[0] = "hcrl_sa"
    datasets[1] = "hcrl_ch"
    datasets[2] = "set_01"
    datasets[3] = "set_02"
    datasets[4] = "set_03"
    datasets[5] = "set_04"

    print ""
    print "================================"
    print "TEACHER_DQN_JOBS=("
    for (i = 0; i < 6; i++) {
      ds = datasets[i]
      if (jobs[ds] != "") {
        printf "  \"%s\"  # %s\n", jobs[ds], ds
      } else {
        printf "  \"NOT_FOUND\"  # %s\n", ds
      }
    }
    print ")"
  }'

echo ""
echo "================================"
echo "Copy the TEACHER_DQN_JOBS array above into student_kd.sh"
echo ""
echo "Note: This script finds currently queued/running teacher DQN jobs"
echo "      Teacher jobs are those WITHOUT '_s_' in the job name"
