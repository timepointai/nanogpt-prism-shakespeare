"""
prism_modal_prior.py — isolated runner for Prior-Fused PRISM (T9 × PRISM) on the
`prism-prior-fusion` branch. Separate Modal Volume + app so its resume-state can't
collide with the other eval volumes. Fire-and-forget by default.

    modal run --detach prism_modal_prior.py --extra "<prism_prior_eval.py flags>"

Fetch the artifact when done, and COMMIT it:
    modal volume get prism-eval-prior nanogpt-prism/results/prior_latest.json ./results/
"""
import modal

app = modal.App("prism-eval-prior")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets")
)

vol = modal.Volume.from_name("prism-eval-prior", create_if_missing=True)  # isolated
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/nanogpt-prism-shakespeare.git"
BRANCH = "prism-prior-fusion"


@app.function(image=image, gpu="L4", volumes={WORK: vol}, timeout=24 * 3600)
def run_eval(extra: str = ""):
    import os
    import subprocess
    import sys
    import time

    vol.reload()
    if not os.path.exists(REPO):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "fetch", "origin", "--quiet"], check=False)
        subprocess.run(["git", "-C", REPO, "reset", "--hard", f"origin/{BRANCH}"],
                       check=False)
    print(f"on branch {BRANCH}:",
          subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip(), flush=True)
    vol.commit()

    cmd = [sys.executable, "-u", "prism_prior_eval.py"] + (extra.split() if extra else [])
    proc = subprocess.Popen(cmd, cwd=f"{REPO}/src", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        if time.time() - last > 60:
            vol.commit(); last = time.time()
    rc = proc.wait()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"prism_prior_eval.py exited {rc}. Resume state on the Volume "
                           f"— re-run to continue.")


@app.local_entrypoint()
def main(gpu: str = "L4", extra: str = ""):
    call = run_eval.with_options(gpu=gpu).spawn(extra=extra)
    print(f"\nLaunched Prior-Fused PRISM run (detached). call id: {call.object_id}")
    print("Watch: modal app list  →  modal app logs <ap-id>")
    print("Fetch when done: modal volume get prism-eval-prior "
          "nanogpt-prism/results/prior_latest.json ./results/")
