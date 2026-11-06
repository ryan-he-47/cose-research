# Stroke representation and page diffusion

The current autoencoder maps a variable-length stroke's geometry and absolute
canvas position into a 48-dimensional latent. The encoder is width 192 with
6 layers and 6 heads. The decoder uses continuous progress tau with FiLM and
DCT-based features to reconstruct points along a stroke. Stroke order and
layout follow the source data; short strokes retain their point counts.

The AE is trained first and then frozen. Text exposure in AE training was
reduced to approximately 5% because text strokes are plentiful and easy to fit;
the intent was to keep them from dominating formation of the latent space.
This AE exposure choice differs from the later prior's eight-task mixture.

The page prior has 100,108,592 trainable parameters, width 768, 13 layers,
12 heads and FFN 2048. Page index uses RoPE; within-page position uses sinusoidal
features concatenated with the latent. The original block width is 16 strokes;
page ends use repeated EOP slots. There is no separate classification head in
this diffusion version. A clean stream conditions noisy blocks, and an x0
diffusion objective predicts frozen-AE coordinates.

Eight tasks cover text continuation (T1), drawing/naming an object (O1/O2),
scene description/drawing (S1/S2), scene questions (S3), selecting by a circle
(S4), and naming a circled object (S5). These are prototype learning tasks on
prepared text, drawings and synthetic scenes.

The current continuation reweights task loss contributions and mixes 4, 8, 8,
and 16-stroke blocks over groups of four updates, with 8-stroke generation as
the default. Its new capability measurements remain pending.
