# Provenance and third-party components

MATE was prepared from a FoAR2 working copy and retains RISE-derived and diffusion-policy components. The main [LICENSE](../../LICENSE) contains the inherited CC BY-NC-SA 4.0 text, original copyright notices, and third-party notices for PyTorch, MinkowskiEngine, Diffusion Policy, ACT, and DETR.

| Component | Included source | License/notices |
| --- | --- | --- |
| MinkowskiEngine 0.5.4 | [dependencies/MinkowskiEngine](../../dependencies/MinkowskiEngine/) | [MIT license](../../dependencies/MinkowskiEngine/LICENSE) |
| PyTorch3D 0.7.9 | [dependencies/pytorch3d](../../dependencies/pytorch3d/) | [License and third-party terms](../../dependencies/pytorch3d/LICENSE) |
| DETR-derived transformer | [policy/transformer.py](../../policy/transformer.py) | Retained copyright header and notices in the main license |

The bundled trees are source snapshots. Native build products and installation metadata are ignored by Git. Upstream PyTorch3D test fixtures and tutorial datasets are excluded from the published repository; the corresponding upstream tests/tutorials may require those assets to be obtained separately. MATE datasets, checkpoints, measured calibration, robot SDKs, and the site adapter are supplied separately under their respective terms.
