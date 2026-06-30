## June 30

Adversary training succeeded. The misclassified images have some noise difference for it to generalize from embedding attack to image attack.

I found there is a difference between the uint8 image and the continuous embedding. Obviously the uint8 image is harder to attack.
And I found that the adversarial example has some kind of unstable generalization on different questions. It is very sensitive to the question.

Built the attack generation on image01 / resized images. Output the adversarial benchmark.

## June 29

Started to design the framework for VLM, including generation, evaluation and attack training.
Identified four stage of images: raw, resized, image01, and pixel_value. Attack happens at image01 stage.

Caveat: I found that the classification accuracy is very sensitive on the prompt. For example, the trailing space.

Successfully built the abstraction in task framework for VLM, ImageClass. Accuracy of LLaVA 1.5 on Imagenette raises to near 1.

Setup on Raven cluster.

## June 28

Decided on the project direction.

We decide to start with zero-shot classification tasks on Oxford-IIIT-Pet.
But it turns out that LLaVA has difficulty classifying Oxford-IIIT-Pet. The accuracy is lower than the low-level CLIP encoder.
After switch to STL-10, and add answer priming, the accuracy increases a lot.