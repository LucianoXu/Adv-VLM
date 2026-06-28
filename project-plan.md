# Project Plan on Adversarial Example Study of Multimodal LLM

Reseach Question:
1. How does the vulnerability against adversarial examples in image encoders generalize to different VLMs?
2. How does the adversarial training of encoders generalize in different VLMs?
3. About the robust training. Which way is better? Pooled embedding or patch token?

Component Choices:
This research is based on the following components.
- Visual Encoder: CLIP ViT-L/14@336
- Baseline: LLaVA-1.5-7B
- LinearAttention: VisualRWKV-7B

Steps:

Attack Phase

1. Find adversarial examples for CLIP (encoder-only attack, encoder-only targeted, end-to-end in VLM)
2. Test the result of adversarial examples on LLaVa and VisualRWKV

Defend Phase

1. Finetune CLIP by adversarial training (standalone, and embedded in LLaVA/VisualRWKV)
2. Test how different trainings enhance robustness

## Background

Adversarial example in MLLM is a hot and well-explored topic. The basic findings are:
1. Adversarial example works.
2. Jailbreak through adversarial image input is also saturated.
3. About the mechanism of advesarial example: research shows that naturally trained models tends to be vulnerable against high-frequency disturbance. But generation of low-frequency noise for adversarial generation is also possible.
4. Encoder contributes mainly to the vulnerability. Attack and robustness training can generalize from standalone encoder to embedded MLLM.
5. The similar phenomenon has been observed on other modalities like audio.