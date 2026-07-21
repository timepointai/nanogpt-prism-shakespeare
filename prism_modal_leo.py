"""
prism_modal_leo.py — isolated runner for Leo's `improvement` branch (via leo-test:
his branch + far corpus + hit==0 guard). Uses a SEPARATE Modal Volume and app so
the third parallel run can't collide with the two live master sweeps on the shared
"prism-eval" volume. Fire-and-forget by default.

    modal run --detach prism_modal_leo.py --extra "<prism_eval.py flags>"
"""
import modal

app = modal.App("prism-eval-leo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets")
)

vol = modal.Volume.from_name("prism-eval-leo", create_if_missing=True)  # isolated
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/nanogpt-prism-shakespeare.git"
BRANCH = "leo-test"


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

    cmd = [sys.executable, "-u", "prism_eval.py"] + (extra.split() if extra else [])
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
        raise RuntimeError(f"prism_eval.py exited {rc}. Resume state on the Volume "
                           f"— re-run to continue.")


@app.local_entrypoint()
def main(gpu: str = "L4", extra: str = ""):
    call = run_eval.with_options(gpu=gpu).spawn(extra=extra)
    print(f"\nLaunched Leo-branch run (detached). call id: {call.object_id}")
    print("Watch: modal app list  →  modal app logs <ap-id>")
    print("Fetch when done: modal volume get prism-eval-leo "
          "nanogpt-prism/results/latest.json ./results/")
