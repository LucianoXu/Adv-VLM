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