# edible

AI-powered wild berry and foraging identification — educational tool only.

> **Safety rule:** when uncertain, reject. A false rejection (annoying the user) is always safer than a false acceptance (bad image reaching the edibility model).

## References

Research consulted during model architecture and training decisions:

**Loss functions & class imbalance**
- [Taming the Tail: Leveraging Asymmetric Loss for Medical Image Long-Tailed Class Imbalance](https://arxiv.org/pdf/2410.04084) — motivation for ASL over focal loss for imbalanced multi-class problems
- [LMFLOSS: A Hybrid Loss For Imbalanced Medical Image Classification](https://arxiv.org/pdf/2212.12741) — comparison of focal, asymmetric, and unified focal losses
- [Two-Stage Fine-Tuning: A Novel Strategy for Learning Class-Imbalanced Data](https://arxiv.org/pdf/2207.10858) — balanced pre-training then imbalanced fine-tuning; informs `--balanced-sampling` design
- [Improving Calibration by Relating Focal Loss, Temperature Scaling, and Properness](https://arxiv.org/html/2408.11598v1) — theoretical grounding for combining focal-family losses with post-hoc temperature scaling

**Confidence calibration**
- [A Comparative Study of Confidence Calibration in Deep Learning](https://arxiv.org/pdf/2206.08833) — survey of calibration methods; supports temperature scaling + label smoothing combination
- [Bin-wise Temperature Scaling (BTS)](https://arxiv.org/pdf/1908.11528) — extension of scalar temperature scaling; baseline for our per-class threshold approach
- [Neural Network Calibration](https://geoffpleiss.com/blog/nn_calibration.html) — Pleiss et al. original temperature scaling write-up

**Hard negative mining & safety-critical vision**
- [Optimizing Detection Reliability in Safety-Critical Computer Vision](https://www.mdpi.com/1424-8220/25/20/6306) — multi-task learning + hyperparameter tuning for safety-critical classifiers; informs toxic-FP early stopping design
- [Hard Negative Sample Mining for Whole Slide Image Classification](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_14) — motivation for boosting sampling frequency of known FP images in future training runs
- [Fine-Grained Hard Negative Mining](https://arxiv.org/pdf/2301.01079) — applying HNM to minority-class detection; directly applicable to ilex_decidua FP reduction

**Transfer learning on small datasets**
- [Fine-Tuning On Small Datasets: How Far Can Pretrained Models Go?](https://aicompetence.org/fine-tuning-on-small-datasets/) — regularization (dropout, weight decay, early stopping) guidance for sub-10k image datasets
- [EfficientNet-B0 for apple leaf disease classification](https://www.nature.com/articles/s41598-025-04479-2) — confirms EfficientNet-B0 + class weighting + augmentation as strong baseline for fine-grained botanical classification