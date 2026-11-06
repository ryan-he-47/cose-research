# Reproduction and asset status

For the earlier GMM stage, start with `configs/r_joint_scratch_20261022.json`,
`src/st/pretrain_model.py` and the existing generation/text-readout modules.
`python -m scripts.demo.demo_pretrain_generation --help` lists the generation
entry point. The [historical gallery](historical_gmm.md) records the actual
checkpoint and readout protocol; those weights are not yet bundled or hosted.

Install `requirements.txt`, then run `python scripts/check_release.py` from the
repository root. `python -m scripts.checks.check_page_diffusion` is the preserved
synthetic CPU check for future/clean-target isolation, clean KV equivalence,
page positions, EOP behavior and loss/gradient consistency. It needs no dataset
or checkpoint. Other development checks may require local research assets.

Use `python -m src.st.train_page_diffusion --help` to inspect training options.
Training needs a data manifest, the matching frozen AE, a recipe and an output
directory. Recorded recipes are in `configs/`; they are not a one-command data
download. Data assembly and training must be planned for the user's own machine.
Set `COSE_REQUIRE_SLURM=1` in environments that require a Slurm job for training.

The release includes source code, recorded research configurations, aggregate
results and a fixed generated panel. The original source is identified by its
commit in `release_manifest.json`. Packaging removes local operational records
and replaces a developer-specific login-node guard with a generic environment
flag; it does not change the model architecture or numerical recipe.

Handwriting libraries, QuickDraw-derived assets, TinyStories material and large
prepared caches are not redistributed here. Dataset-specific terms remain
upstream. Matching AE/prior checkpoints are also not yet published; their
status is listed in `assets/manifest.json`. Reproducing the reported trained
outputs requires those exact assets or a new training run. CPU source checks
remain independently runnable without them.
