# student_l1b_c_featkd — selected deployment student

This is the width-16 NAFNet student (1.7 M parameters) selected for deployment
in the final report (checkpoint epoch 99, chosen by validation CLIP ViT-B/32
cosine similarity 0.8585).

Heavy run outputs (checkpoints, ONNX/LiteRT/HEF artefacts, TensorBoard logs)
are gitignored and stored in Weights & Biases; they are available to the
supervisor and examiners on request.

Pointers:

- Training config: `configs/student_l1b_c_featkd.yaml`
- Training command (from `training/`): `python scripts/train.py --config configs/student_l1b_c_featkd.yaml`
- Distillation objective: teacher-output L1 (50) + GT L1 (5) + DINOv2/CLIP/
  ResNet-50/SegFormer/DepthAnything feature losses + teacher feature-KD (0.5)
- Evaluation rows: `experiments/eval_full_v4_canonical/summary.csv`
  (model `student_l1b_c`, epoch 99) and the INT8 export under
  `experiments/eval_full_v5_int8/`
