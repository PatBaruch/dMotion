# Historical model experiments

These notes retain earlier evidence and rationale. The active workflow is
[YOLO26m in the README](../../README.md).

- [3 October training experiment](TRAINING_RUN_2026-10-03.md): older YOLO26 Nano cash detector.
- [YOLOE proposal](YOLOE_FINETUNING.md): historical reference/fine-tuning design.

Old commands, paths, confidence settings, and dataset counts are historical.
YOLO-World, YOLOE, Grounding DINO, and text-encoder integrations are retired from
the maintained application. Recover their source through Git history if needed.

Do not delete an old output directory merely because of its model name. Current
YOLO26m runs can refer to reviewed datasets exported by earlier YOLOE experiments.
Keep manifests, reports, split maps, hashes, evaluation helpers and active-run
dependencies together until their consumers have finished.
