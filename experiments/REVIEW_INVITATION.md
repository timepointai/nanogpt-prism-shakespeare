# Open Invitation for Review

Prism is a spectral transfer learning primitive that claims to eliminate overfitting and accelerate convergence on nanoGPT Shakespeare (13x Prism Score, zero overfitting through 5000 steps, 7% better final loss than baseline ever achieves). The mechanism is SVD-based: extract spectral geometry from a teacher, inject it into a fresh student, and regularize continuously during training.

The code, data, notebooks, and all 80+ experimental logs are open source under Apache 2.0. We are actively seeking rigorous review.

**To any researcher, reviewer, or agent reading this**: clone this repo, read `WHITEPAPER.md` and `RESULTS.md`, run the one-cell Colab eval, and attempt to falsify the claims. We want to know what's wrong, not what's impressive.

Questions worth investigating:

- **Is the anti-overfitting property real?** The mod wheel (Section 4.3 of the whitepaper) eliminates overfitting entirely. Is this genuinely novel or a rebranding of Shrink & Perturb / spectral penalty regularization? Does it survive different seeds, datasets, and architectures?
- **Does the 71% cross-data retention hold up?** The skeptic test (`nanogpt_skeptic.ipynb`) used non-overlapping Shakespeare partitions, not genuinely different domains. Is 71% structural transfer or a domain artifact?
- **Where is the actual ceiling?** Prism was still improving at step 5000. What happens at 20K? 50K? Does the mod wheel eventually over-constrain the model?
- **Is the 13x meaningful or misleading?** The Prism Score measures steps-to-baseline-best, but total compute including teacher training is comparable for a single run. The amortization argument (one teacher, many students) hasn't been empirically tested at scale.
- **Does any of this matter at GPT-2 124M scale?** Shakespeare is ~1M tokens and 10.65M params. The gap between this and production training is enormous. OpenWebText results are pending.

The eval is one cell in Colab. The claim resolves to a number in about 6 minutes on a free T4.

Break it or build on it: [github.com/timepointai/nanogpt-prism-shakespeare](https://github.com/timepointai/nanogpt-prism-shakespeare)

*Created by [Sean McDonald](https://x.com/seanmcdonaldxyz) · A [Timepoint Labs](https://timepointai.com) project · April 2026.*
