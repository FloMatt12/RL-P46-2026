#!/usr/bin/env bash
# Runs run_envs.py inside a RenkuLab job and saves the results with git, because a job's files are
# deleted when it ends. Every SYNC_MINUTES (default 30) and at the end it commits results/**/*.npy and
# manifest.json and pushes them. Logs are pushed once at the end (compressed).
# At start it restores the results already pushed, so a restarted job skips finished runs.
#
# Usage:  bash renku_job.sh <target> <run_envs.py arguments...>
#   <target> = a branch name    -> results go to that branch of this repo (needs push access to this repo)
#            = a directory      -> another cloned repo (e.g. a second code repository in the Renku project);
#                                  results go to the branch it has checked out
# Renku job commands (repos are cloned into /home/renku/work/<repo name>/, the job starts in /home/renku/work):
#   ["bash", "RL-P46-2026/renku_job.sh", "rl-p46-results", "--workers", "14", "--tag", "paper_1M"]
#   ["bash", "RL-P46-2026/renku_job.sh", "part3-results", "--workers", "14", "--tag", "paper_1M"]
set -u

TARGET="$1"; shift
REPO="$(cd "$(dirname "$0")" && pwd)"
SYNC_MINUTES="${SYNC_MINUTES:-30}"

if [ -d "$TARGET/.git" ]; then
	# --- separate results repo ---
	WT="$(cd "$TARGET" && pwd)"
	BRANCH="$(git -C "$WT" rev-parse --abbrev-ref HEAD 2>/dev/null)" \
		|| { echo "$TARGET has no commits yet; create it on GitHub with a README." >&2; exit 1; }
	cd "$REPO"
	mkdir -p "$REPO/results"
	[ -d "$WT/results" ] && cp -r "$WT/results/." "$REPO/results/"   # restore finished runs
else
	# --- results branch in this repo (checked out next to the code clone) ---
	BRANCH="$TARGET"
	WT="$REPO/../results-worktree"
	cd "$REPO"
	if git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" 2>/dev/null; then
		git worktree add -q -B "$BRANCH" "$WT" "origin/$BRANCH"
		mkdir -p "$REPO/results"
		[ -d "$WT/results" ] && cp -r "$WT/results/." "$REPO/results/"   # restore finished runs
	else
		git worktree add -q --detach "$WT"
		(cd "$WT" && git checkout -q --orphan "$BRANCH" && git rm -rq --cached . && git clean -fdxq)
	fi
fi

push_results() {  # $1 = commit message
	mkdir -p "$WT/results" "$REPO/results"
	(cd "$REPO/results" && find . \( -name '*.npy' -o -name 'manifest.json' -o -name '*.log.tar.gz' \) -print0 \
		| xargs -0 -r cp --parents -t "$WT/results/")
	cd "$WT"
	git add -A results
	git commit -q -m "$1" >/dev/null || true   # nothing new -> nothing to commit
	for attempt in 1 2 3; do
		git pull -q --rebase origin "$BRANCH" 2>/dev/null   # another job may have pushed to the same branch
		git push -q origin "HEAD:refs/heads/$BRANCH" && { cd "$REPO"; return 0; }
		sleep 30
	done
	echo "WARNING: push to $BRANCH failed" >&2
	cd "$REPO"; return 1
}

# Fail early (before hours of compute) if the job can't push.
(cd "$WT" && git commit -q --allow-empty -m "Job started: run_envs.py $*")
push_results "Job started" || { echo "Cannot push to $BRANCH; stopping." >&2; exit 1; }

python -u "$REPO/run_envs.py" "$@" &
PID=$!
trap 'push_results "Job stopped early (SIGTERM)"; exit 143' TERM
while kill -0 "$PID" 2>/dev/null; do
	for _ in $(seq "$SYNC_MINUTES"); do kill -0 "$PID" 2>/dev/null && sleep 60; done
	kill -0 "$PID" 2>/dev/null && push_results "Results update ($(date -u +%Y-%m-%dT%H:%MZ))"
done
wait "$PID"; STATUS=$?

for logs in "$REPO"/results/*/logs; do
	[ -d "$logs" ] && tar -czf "$logs.log.tar.gz" -C "$(dirname "$logs")" logs
done
push_results "Job finished (run_envs.py exit code $STATUS)"
exit "$STATUS"
