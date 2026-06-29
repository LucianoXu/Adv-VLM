## June 29

Started to design the framework for VLM, including generation, evaluation and attack training.
Identified four stage of images: raw, resized, image01, and pixel_value. Attack happens at image01 stage.

Caveat: I found that the classification accuracy is very sensitive on the prompt. For example, the trailing space. Also the currenct mean log-likelyhood approach has a bais on long, predictable sequence.

## June 28

Decided on the project direction.

We decide to start with zero-shot classification tasks on Oxford-IIIT-Pet.
But it turns out that LLaVA has difficulty classifying Oxford-IIIT-Pet. The accuracy is lower than the low-level CLIP encoder.
After switch to STL-10, and add answer priming, the accuracy increases a lot.