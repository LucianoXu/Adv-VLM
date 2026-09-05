#!/bin/bash
# Submit the AID-2026 experiment pipeline to Slurm with the right dependencies.
#
# Every number in the paper comes from one consistent run of this pipeline. The
# old adversarial datasets were never committed (dataset/ is gitignored), so they
# have to be regenerated -- and a regenerated attack is not bit-identical to the
# original (Adam plus non-deterministic CUDA kernels), which is why the whole
# table is re-derived rather than partly reused.
#
# Stages, and what depends on what:
#
#   A  attacks-cls    the four main adversarial datasets + the five budget-sweep ones
#   B  eval-cls       clean baselines, the cross-eval matrix, the budget sweep
#   C  defense-cls    input-transformation defence on classification    (needs A)
#   D  attacks-jb     universal jailbreak images on LLaVA and VisualRWKV
#   E  gen-jb         the 2x2 transfer matrix generations               (needs D)
#   F  defense-jb     defended generations on the LLaVA jailbreak image (needs D)
#
# B/C run off A; E/F run off D; the two chains are independent of each other and
# are submitted in parallel.
#
# Grading (the OpenAI judge) is NOT here: compute nodes have no internet. Run it
# on a login node afterwards -- see the tail of this script for the commands.
#
# Usage:
#   scripts/submit_pipeline.sh                 # everything
#   scripts/submit_pipeline.sh A B             # only those stages
#   DRYRUN=1 scripts/submit_pipeline.sh        # print the sbatch calls, submit nothing

set -euo pipefail
cd "$(dirname "$0")/.."

SB=scripts/run_config.sbatch
DRYRUN="${DRYRUN:-}"

submit() {   # submit <name> <config> <timelimit> [dependency-jobid]
  local name="$1" cfg="$2" tl="$3" dep="${4:-}"
  local args=(--job-name="$name" --time="$tl")
  [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep")

  if [[ ! -f "$cfg" ]]; then
    echo "SKIP $name: no such config $cfg" >&2
    return 1
  fi
  if [[ -n "$DRYRUN" ]]; then
    # the job id goes to stdout (callers capture it); everything human-readable
    # goes to stderr, or command substitution would eat it
    echo "  would run: sbatch ${args[*]} $SB $cfg" >&2
    echo "DRYRUN-$name"
    return 0
  fi
  local jid
  jid=$(sbatch --parsable "${args[@]}" "$SB" "$cfg")
  echo "  $name -> job $jid${dep:+ (after $dep)}" >&2
  echo "$jid"
}

want() {
  [[ $# -eq 0 ]] && return 0
  local s
  for s in "${STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done
  return 1
}

# Configs belonging to each stage, so preflight validates only what is being
# submitted. Validating the whole pipeline would fail forever once an early stage
# has run and legitimately owns its output directories.
stage_configs() {
  case "$1" in
    A) echo "configs/clip-llava-attack-imagenette.yaml configs/budget-sweep-attack.yaml" ;;
    B) echo "configs/xeval-clean.yaml configs/xeval.yaml configs/xeval-rwkv.yaml configs/budget-sweep-eval.yaml" ;;
    C) echo "configs/defense-classify.yaml" ;;
    D) echo "configs/jailbreak-attack-llava.yaml configs/jailbreak-attack-rwkv.yaml" ;;
    E) echo "configs/jailbreak-gen.yaml" ;;
    F) echo "configs/defense-jailbreak.yaml" ;;
  esac
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(A B C D E F)

# Gate on the config validator. The most common way this pipeline dies is a stale
# output_dir: run.py calls os.makedirs(..., exist_ok=False), so one pre-existing
# directory aborts a task -- possibly hours into a queued job. preflight also
# checks task types, required keys, input paths and defence blocks.
# Set SKIP_PREFLIGHT=1 only if you have a reason.
if [[ -z "${SKIP_PREFLIGHT:-}" && -z "$DRYRUN" ]]; then
  PF_CONFIGS=()
  for st in "${STAGES[@]}"; do
    for c in $(stage_configs "$st"); do
      [[ -f "$c" ]] && PF_CONFIGS+=("$c")
    done
  done
  echo ">> preflight (${#PF_CONFIGS[@]} config(s) for stage(s) ${STAGES[*]})" >&2
  if ! python scripts/preflight.py "${PF_CONFIGS[@]}" > /tmp/advvlm-preflight.$$ 2>&1; then
    echo "" >&2
    grep -E "^  - |^[0-9]+ tasks|error\(s\)" /tmp/advvlm-preflight.$$ >&2 ||       cat /tmp/advvlm-preflight.$$ >&2
    echo "" >&2
    echo "!! preflight failed -- nothing submitted." >&2
    echo "!! stale output dirs are the usual cause; archive the previous run, e.g." >&2
    echo "!!   mkdir -p results/_original-run && mv results/xeval* results/_original-run/" >&2
    rm -f /tmp/advvlm-preflight.$$
    exit 1
  fi
  tail -3 /tmp/advvlm-preflight.$$ >&2
  rm -f /tmp/advvlm-preflight.$$
fi

echo ">> submitting stages: ${STAGES[*]}" >&2
A_JID=""; D_JID=""

# ---- classification chain -------------------------------------------------
if want A; then
  echo ">> stage A: craft adversarial datasets" >&2
  # 4 main datasets (CLIP/LLaVA x image01/resized) then the 5 budget variants;
  # one job so the budget attacks cannot start before the main ones have the GPU
  A1=$(submit advvlm-attack-main configs/clip-llava-attack-imagenette.yaml 04:00:00)
  A_JID=$(submit advvlm-attack-budget configs/budget-sweep-attack.yaml 02:00:00 "$A1")
fi

if want B || want C; then
  A_JID=$(require_upstream "$A_JID" advvlm-attack-budget B)
  [[ -z "$A_JID" ]] && A_JID=$(require_upstream "" advvlm-attack-main B)
  if [[ -z "$A_JID" ]]; then
    echo "  note: no stage A job in this submission or the queue; assuming its" >&2
    echo "  datasets are already on disk (preflight checks their paths)." >&2
  fi
fi

if want B; then
  echo ">> stage B: evaluations" >&2
  submit advvlm-xeval-clean configs/xeval-clean.yaml     08:00:00 "$A_JID" >/dev/null
  submit advvlm-xeval        configs/xeval.yaml          12:00:00 "$A_JID" >/dev/null
  # 12 VisualRWKV tasks over 1024 images, 10 candidate labels each, and RWKV
  # recomputes the full sequence per candidate -- this is the long one
  submit advvlm-xeval-rwkv   configs/xeval-rwkv.yaml     23:00:00 "$A_JID" >/dev/null
  submit advvlm-budget-eval  configs/budget-sweep-eval.yaml 08:00:00 "$A_JID" >/dev/null
fi

if want C; then
  echo ">> stage C: defence on classification" >&2
  submit advvlm-defense-cls  configs/defense-classify.yaml 08:00:00 "$A_JID" >/dev/null
fi

# When a downstream stage is submitted WITHOUT its upstream stage in the same
# invocation, the *_JID variables are empty and the job would go in with no
# dependency -- eligible to start immediately against inputs that do not exist yet.
# That happened once (E/F resubmitted alone after stage D was relaunched). So look
# the upstream job up by name in the live queue and refuse rather than guess.
find_job() {   # find_job <job-name> -> most recent matching job id, or empty
  squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v n="$1" '$2 == n {print $1}' | sort -n | tail -1
}

require_upstream() {   # require_upstream <var-value> <job-name> <stage-letter>
  local jid="$1" name="$2" letter="$3"
  if [[ -n "$jid" ]]; then echo "$jid"; return 0; fi
  local found
  found=$(find_job "$name")
  if [[ -n "$found" ]]; then
    echo "  note: stage $letter not in this submission; chaining to live job $found ($name)" >&2
    echo "$found"
    return 0
  fi
  echo "" 
  return 0
}

# ---- jailbreak chain ------------------------------------------------------
if want D; then
  echo ">> stage D: universal jailbreak attacks" >&2
  D1=$(submit advvlm-jb-attack-llava configs/jailbreak-attack-llava.yaml 08:00:00)
  # the VisualRWKV attack config is written by the RWKV-attack work; skip cleanly
  # if it is not in place yet and submit stage D again once it is
  if [[ -f configs/jailbreak-attack-rwkv.yaml ]]; then
    D_JID=$(submit advvlm-jb-attack-rwkv configs/jailbreak-attack-rwkv.yaml 08:00:00 "$D1")
  else
    echo "  note: configs/jailbreak-attack-rwkv.yaml missing -- LLaVA attack only" >&2
    D_JID="$D1"
  fi
fi

if want E || want F; then
  # stage D may have been submitted in an earlier invocation
  D_JID=$(require_upstream "$D_JID" advvlm-jb-attack-rwkv E)
  [[ -z "$D_JID" ]] && D_JID=$(require_upstream "" advvlm-jb-attack-llava E)
  if [[ -z "$D_JID" ]]; then
    echo "!! stage E/F requested but no stage D job found, in this submission or in" >&2
    echo "!! the queue. Submitting them now would start against inputs that do not" >&2
    echo "!! exist. Submit D first, or pass D alongside E/F." >&2
    exit 1
  fi
fi

if want E; then
  echo ">> stage E: jailbreak transfer generations" >&2
  submit advvlm-jb-gen configs/jailbreak-gen.yaml 23:00:00 "$D_JID" >/dev/null
fi

if want F; then
  echo ">> stage F: defended jailbreak generations" >&2
  submit advvlm-defense-jb configs/defense-jailbreak.yaml 12:00:00 "$D_JID" >/dev/null
fi

cat >&2 <<'EOF'

>> submitted. watch with:  squeue -u yinxu -o "%.10i %.24j %.2t %.10M %.10L %R"
>> logs:                   scripts/logs/<job-name>_<jobid>.{out,err}

>> AFTER the GPU stages finish, grade on a LOGIN node (needs internet + OPENAI_API_KEY):
     source .venv/bin/activate
     python -c "from src import run; run('configs/jailbreak-grade.yaml')"
     python -c "from src import run; run('configs/defense-jailbreak-grade.yaml')"

>> then aggregate (each writes a .tex fragment for the paper):
     python scripts/aggregate_jailbreak.py     # 2x2 matrix, Wilson CIs, McNemar, judge kappa
     python scripts/aggregate_budget.py        # transfer-vs-budget curve (pgfplots)
     python scripts/aggregate_defense.py       # defence table (also cross-checks the
                                               # undefended row against the budget sweep)

>> and regenerate the figures, which currently come from the deleted datasets:
     python scripts/make_figures.py \
       --samples results/clip-imagenette-attack-resized/dataset \
       --jailbreak results/jailbreak/attack-llava-s42
EOF
