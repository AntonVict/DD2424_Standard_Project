# Default Project 1 - DD2424, 2025: Exploring Transfer Learning

This project focuses on exploring the concept of transfer learning, a common use case in deep learning. [cite: 1, 2] The core idea is to download a pre-trained model and adapt it to a new dataset. [cite: 2]

## Phases of Development

This project is structured into a basic component and several extensions for achieving higher grades.

### Phase 1: Basic Project (To Achieve Grade E)

The initial phase involves setting up the environment, downloading necessary components, and performing binary and multi-class classification tasks.

**Steps:**

1.  **Download a Pre-trained Modern ConvNet:**
    * Choose a network such as ResNet18, ResNet34, or similar.
2.  **Download the Dataset:**
    * The recommended dataset is The Oxford-IIIT Pet Dataset. [cite: 26] You are free to use another dataset, but it should have comparable or greater difficulty. [cite: 26, 27]
3.  **Binary Classification (Dog vs. Cat):**
    * Replace the final layer of the pre-trained ConvNet to solve the binary classification problem of recognizing pictures of dogs versus cats.
    * Fine-tune the replaced final layer using the Pet Dataset's training data. [cite: 5]
    * Use Adam or NAG optimizer. [cite: 5]
    * Aim for high performance (≥99% test accuracy). [cite: 6]
    * **Important Considerations:**
        * Check the spatial size of input your pre-trained network can handle. [cite: 7] Default ResNet architectures use a global average pooling layer, making them flexible to input image size, but ensure down-sampling doesn't cause feature maps to disappear. [cite: 8, 9]
        * If images are too small, resize them to the smallest acceptable size to minimize computational effort, maintaining the aspect ratio by rescaling the shortest side. [cite: 10, 11]
4.  **Multi-class Classification (Breed Recognition):**
    * Replace the final layer to have 37 outputs (for the Pet Dataset). [cite: 13]
    * Fine-tune more of the network, not just the replaced final layer, as this is a harder problem. [cite: 14]
    * Explore the following fine-tuning issues, using a validation set and computational budget to decide when to end training: [cite: 15]
        * **Strategy 1: Fine-tune *l* layers simultaneously:** Fine-tune the last *l* layers (plus the classification layer) from the start. Experiment with *l*=1, then *l*=2, *l*=3, up to *L* (defined by available compute and when adding more layers yields minimal/no change). [cite: 16, 17, 18]
        * **Strategy 2: Gradual un-freezing:** Fine-tune in stages, starting with the last few layers and progressively unfreezing earlier ones. [cite: 19, 20]
        * Compare the final performance and training time for these two strategies. [cite: 21]
        * Investigate the use of different learning rates and/or learning rate schedulers for different layers. [cite: 22]
        * Assess the benefit of applying data augmentation (flip, small rotations, crops, small size scaling) and L2 regularization. [cite: 23]
        * Examine the effect of fine-tuning or not fine-tuning batch-norm parameters and updating batch mean/standard deviation estimates on the new dataset. [cite: 24]
    * Aim for a final test accuracy of around 95% (this is a guideline). [cite: 25]
5.  **Fine-tuning with Imbalanced Classes:**
    * Investigate what happens with imbalanced classes. For example, use only 20% of training images for each cat breed. [cite: 28]
    * Observe the final test performance on classes with limited data when using normal cross-entropy loss. [cite: 29]
    * Try strategies like weighted cross-entropy and/or over-sampling of minority classes to compensate. [cite: 30]

### Phase 2: Extensions for Higher Grades

Once the basic project is thoroughly explored, you can add extensions to aim for higher grades.

#### Extensions for Grade D/C [cite: 32]

Investigate if performance can be improved by exploring the following:

* **Deeper Networks:**
    * Explore using deeper networks than in the basic project. [cite: 33]
    * Does it help? Is it trickier to fine-tune? Do earlier layers need fine-tuning? [cite: 33, 34]
    * Might you need to change the optimizer (e.g., AdamW) or perform more L2 regularization? [cite: 35]
* **Catastrophic Forgetting:**
    * Explore the concept of catastrophic forgetting, where a network fine-tuned on a new dataset loses knowledge from the original dataset. [cite: 36, 37]
    * Consider your cat vs. dog classifier as the pre-trained network.
    * Use a dataset distinct from ImageNet and the Pets dataset, such as the 102 Category Flower Dataset. [cite: 38]
    * Attempt to induce catastrophic forgetting by "aggressively" fine-tuning the cat vs. dog network on this new dataset (may require a long training run). [cite: 40]
    * Ensure good performance on the new dataset. Then, retrain the classification layer for the Dog vs. Cat problem on the newly fine-tuned network and check if the original performance can be regained. [cite: 41]
* **Batch Norm Fine-tuning:**
    * Fine-tune only the batch norm mean and standard deviation, keeping other layer weights (except the final layer) frozen, and see if results improve. [cite: 42]
* **Sophisticated Data Augmentations:**
    * Add more advanced data augmentation techniques like random erasing, CutMix, or MixUp to help with regularization. [cite: 43]

**Note on Grading (D/C):** The depth of exploration matters. A thorough investigation of one extension (e.g., CutMix) can be equivalent to exploring many superficially. [cite: 45, 46] Ensure any custom extensions are vetted through your project proposal. [cite: 48]

#### Extensions for Grade B/A [cite: 49]

Explore one or more of the following advanced topics (you do not need to complete the D/C extensions first):

* **Semi-Supervised Learning:**
    * Explore using unlabelled data when labelled training data is limited.
    * Decrease the percentage of labelled training data used during fine-tuning (e.g., 100%, 50%, 10%, 1%) and record the performance drop. [cite: 50]
    * Investigate if using a small percentage of labelled data plus the rest as unlabelled data can achieve similar performance to using the full labelled dataset. [cite: 51]
    * Focus on one approach, such as consistency regularization or pseudo-labelling. Refer to resources like "Learning with not Enough Data Part 1: Semi-Supervised Learning" for an overview. [cite: 53, 54]
* **Vision Transformers (ViTs):**
    * Explore pre-trained ViTs (e.g., from Hugging Face). [cite: 55]
    * Fine-tune a ViT for image classification, building on the basic project. [cite: 55]
    * The report could highlight advantages/disadvantages and the difficulty of fine-tuning a ViT compared to a ResNet. [cite: 56]
* **LoRA (Low-Rank Adaptation) Layers:**
    * Address the challenge of fine-tuning very large pre-trained models with limited compute and memory. [cite: 58, 59]
    * Explore adding LoRA layers to the classification pre-trained model from the first part of the project. [cite: 60, 61]
    * Consider choosing a bigger version of the network you originally tried (if computationally feasible). [cite: 62]
    * Grading will be based on accurate implementation and informative experiments. [cite: 63]
* **Masked Fine-tuning:**
    * Explore a recent parameter-efficient fine-tuning technique where only a small subset of network parameters are updated. [cite: 64, 65]
    * A mask is applied at the parameter update step. [cite: 67]
    * Build on the first part of the project. [cite: 68]
    * Grading will focus on accurate implementation and thoughtful experiments comparing the parameter selection process to baseline methods. [cite: 69]
* **Network Compression:**
    * Once a pre-trained network is well-trained on the new dataset, investigate extracting a version with lower memory and/or compute requirements while maintaining similar performance. [cite: 70, 71]
    * Explore common approaches like pruning network weights and/or quantizing activations/weights. [cite: 72]
    * Refer to resources like "Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding." [cite: 73]

**Note on Grading (B/A):** You are encouraged to come up with your own extensions, but get them vetted through your project proposal. [cite: 74] If aiming for an A, most of the project report should be devoted to the extension. [cite: 75]

## Dataset

* **Primary:** The Oxford-IIIT Pet Dataset.
* **Alternative:** Other datasets of comparable or greater difficulty are acceptable. [cite: 26, 27]
* **For Extensions:** 102 Category Flower Dataset (for Catastrophic Forgetting). [cite: 38]

## Key Technologies/Libraries

* PyTorch (specifically for fine-tuning TorchVision models). [cite: 3]
* Pre-trained ConvNets (e.g., ResNet series).
* Optimizers (Adam, NAG). [cite: 5]
* Optionally: FastAI, Hugging Face Transformers (for ViTs). [cite: 4, 55]

## General Notes

* Tutorials for reference:
    * FINETUNING TORCHVISION MODELS. [cite: 3]
    * FastAI Computer Vision tutorial. [cite: 4]
* For the basic assignment, report the main results and put more extensive fine-tuning results in the appendix. [cite: 76]
* If you pursue an extension from E to A, the majority of the project report should focus on the extension. [cite: 75]

---

This README provides a structured overview of the project, its phases, and requirements based on the provided document. Remember to consult the original project description for full details and any updates.