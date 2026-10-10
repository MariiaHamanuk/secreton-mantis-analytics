#!/bin/zsh
# one line a machine: load, memory, and the episodes kept by tag
P=${SBF_GCP_PROJECT:?set the cloud project}
typeset -A Z; Z=(sbf-s5 us-west1-a sbf-s7 us-west4-a sbf-s8 us-west4-a sbf-s9 us-west4-a sbf-s10 europe-west1-b sbf-s11 europe-west1-b)
for m in ${(ko)Z}; do
  ( gcloud compute ssh $m --project=$P --zone=${Z[$m]} --quiet --command='cd ~/repo; echo load $(cut -d" " -f1 /proc/loadavg) mem $(free -g | awk "NR==2{print \$3}")G; for d in $(ls -d outputs/frontier_lab/scen/play/*/ 2>/dev/null); do n=$(ls $d 2>/dev/null | grep -c pkl); [ $n -gt 0 ] && echo "$(basename $d | sed s/_444//):$n"; done; .venv/bin/python -c "
import pickle,glob
for f in sorted(glob.glob(\"outputs/hazard_lab/play/*.pkl\")):
    try: print(f.split(\"/\")[-1][:-8]+\":\"+str(len(pickle.load(open(f,\"rb\")))))
    except Exception as e: print(f, \"busy\")
"; ls outputs/logs | grep DONE' 2>&1 | grep -v "^Warning\|^$" | tr '\n' ' ' | sed "s/^/$m: /"; echo ) &
done; wait | sort
date
