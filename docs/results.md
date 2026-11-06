# Recorded prototype results

## Additional training

The no-auxiliary diffusion model first completed 360,000,013 successful inputs.
An additional 1,000,000,004 successful inputs brought it to 1,360,000,017. The
comparison below precedes the subsequent task-rebalancing continuation.

On 512 paired synthetic scenes, selecting the named object with a circle
improved from 41.0% to 50.8%. The paired difference was +9.8 percentage points,
with a recorded 95% interval of +6.0 to +13.3 points.

## Conditional use

At high noise, removing the drawing while retaining the question increased
O2 object-naming error by 68.1% after continuation, compared with 4.0% before.
For S4 selection, the increase changed from 28.9% to 94.3%. These diagnostics
indicate stronger conditional use in the measured tasks; they are denoising
error changes, not generated-answer accuracy.

S3 scene questions remain weakly dependent on visual context: removing graphics
increased high-noise error by 4.6%. In the 460-example answer-start S5 subset,
substituting the wrong drawing increased error by only 1.6%. Clear handwriting
can still name the wrong object.

## Generated history

Four-block rollouts on 128 sources per task were compared with true history.
Among trajectories still active at block four, reference-target denoising error
rose to 2.38x for T1 and 3.05x for S1. Decoding and re-encoding the same generated
curves changed those errors by only about 0.24% and 0%. The recorded analysis
therefore focused subsequent work on content paths and task contributions.

Long text still contains invalid-looking letter sequences and drifts in content;
complex drawings have structural errors. The [fixed panel](gallery.md) retains
failures as well as successes. These measurements concern the stated synthetic
tasks and do not establish general graphical reasoning.

Source identity: `bd8dd8c89d644af31e2515900afa5f1b1663f294`, recorded report
`docs/records/r/2026-11-05_page_diffusion_r_diagnostics.md` in the development
repository. Compact data provenance is in [the evidence record](evidence/results.json).
