"""
prism_modal.py — Run the Prism eval as a headless Modal GPU job.

Why this exists: Colab is a browser-tethered kernel and drops mid-run. Modal
runs the job server-side on a rented GPU with nothing to babysit — pay per
second, auto-teardown, and a persistent Volume so a preempted run resumes
instead of restarting.

One-time setup (yours — I don't handle the token):
    pip install modal
    python3 -m modal setup          # opens a browser, authenticates once

Run it (from your local clone of this repo):
    modal run prism_modal.py                       # recipe, seeds 1337,1338,1339
    modal run --detach prism_modal.py              # survives your laptop closing
    modal run prism_modal.py --seeds 1337          # one seed (a sample, not a result)
    modal run prism_modal.py --gpu A10G            # override the GPU

What it does:
    - Clones (then `git pull`s) this repo ONTO a persistent Volume, so the eval's
      resume dirs (.prism_runs / .prism_cache / results / checkpoints) survive a
      container recycle. A re-run skips every finished stage.
    - Streams the eval's live output to your terminal and commits the Volume
      every ~60s, so even a hard kill loses at most a minute of the stage in
      flight — never a finished 110-minute run.
    - On success, writes the final artifact into your LOCAL results/ so you can
      commit it. (If the run is interrupted before returning, the artifact is
      still on the Volume: `modal volume get prism-eval results ./pulled`.)

The model is tiny (10.65M params), so it can't saturate a big GPU — L4 is the
sweet spot (cheap, modern, schedules fast). A100/H100 cost more for no speedup.
"""
import modal

app = modal.App("prism-eval")

# torch's default PyPI wheel bundles CUDA; Modal supplies the driver.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "transformers", "tiktoken", "datasets")
)

# Persistent storage for the repo working tree + all resume state + artifacts.
vol = modal.Volume.from_name("prism-eval", create_if_missing=True)
WORK = "/work"
REPO = f"{WORK}/nanogpt-prism"
REPO_URL = "https://github.com/timepointai/nanogpt-prism-shakespeare.git"


@app.function(image=image, gpu="L4", volumes={WORK: vol}, timeout=24 * 3600)
def run_eval(method: str, seeds: str, teacher_steps: int, student_steps: int):
    import os
    import subprocess
    import sys
    import time

    # Code lives on the Volume so working dirs persist. Fresh clone, else pull.
    if not os.path.exists(REPO):
        print(f"cloning {REPO_URL} → {REPO}", flush=True)
        subprocess.run(["git", "clone", REPO_URL, REPO], check=True)
    else:
        print("repo present on Volume — pulling latest code", flush=True)
        subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)
    vol.commit()

    # Run the resumable eval as a child; echo its live stream and snapshot the
    # Volume every ~60s so a preemption keeps whatever finished.
    proc = subprocess.Popen(
        [sys.executable, "-u", "prism_eval.py",
         f"--method={method}", f"--seeds={seeds}",
         f"--teacher_steps={teacher_steps}",
         f"--student_steps={student_steps}"],
        cwd=f"{REPO}/src", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    last_commit = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        if time.time() - last_commit > 60:
            vol.commit()
            last_commit = time.time()
    rc = proc.wait()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"prism_eval.py exited {rc}. Resume state is on the "
                           f"Volume — re-run to continue where it stopped.")

    # Hand the final artifact back so the local entrypoint can save it for commit.
    art = f"{REPO}/results/latest.json"
    name = os.path.basename(os.path.realpath(art))
    with open(art) as f:
        return {"name": f"recipe_{name}" if name == "latest.json" else name,
                "content": f.read()}


@app.local_entrypoint()
def main(gpu: str = "L4", method: str = "recipe", seeds: str = "1337,1338,1339",
         teacher_steps: int = 2000, student_steps: int = 5000):
    import pathlib

    # Bind the requested GPU at call time so --gpu works without editing code.
    result = run_eval.with_options(gpu=gpu).remote(
        method=method, seeds=seeds,
        teacher_steps=teacher_steps, student_steps=student_steps)

    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    dest = out / result["name"]
    dest.write_text(result["content"])
    print(f"\nArtifact written locally → {dest}")
    print("COMMIT THIS FILE — it is the evidence for any claim you publish.")
