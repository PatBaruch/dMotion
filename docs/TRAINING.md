# YOLO26m training

The complete supported workflow is in [README: collect and review](../README.md#collect-and-review-a-dataset)
and [README: train and evaluate](../README.md#train-yolo26m-and-evaluate-the-candidate).
Those sections cover first-time setup, checkpoint selection, labeling controls,
recording groups, draft review, training options, output paths, validation, test,
and local/externally trained checkpoints.

YOLO26m is the only maintained model family. The generic `models/yolo26m.pt`
initializes training; it is not the trained cash detector. Runs save candidates
under `outputs/yolo26m-training/` without replacing the live checkpoint.

Historical YOLO26 Nano results and the proposed YOLOE route are retained in
[archive](archive/README.md). They describe old experiments, not current commands.
