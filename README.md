# Is Adversarial Training Really a Silver Bullet for Mitigating Data Poisoning?

This repository contains the PyTorch implementation of our ICLR 2023 paper titled "[**Is Adversarial Training Really a Silver Bullet for Mitigating Data Poisoning?**](https://openreview.net/pdf?id=zKvm1ETDOq)".


## Usage

### Step 1: Train the reference model using clean data

 ```
 python train_reference_model.py --robust_eps 4 --reference_path ./reference_model
 ``` 
 You can control the robustness of the reference model by ajusting ```--robust_eps``` parameter.
 The reference model will be saved at ```--reference_path```.
 This file can also be used to evaluate the attack performance, with poisoned data as input.

### Step 2: Calculate the centroid for each class

 ```
 python get_centroid.py --centroid_path ./centroid
 ```
 The centroid will be saved at ```--centroid_path```

### Step 3: Generate poisons

 ```
 python poison_generate.py --eps 8 --recipe push
 ```
 The poison budget can be controlled by adjusting ```--eps```.
 You can select the poisoning method by setting ```--recipe push``` corresponding to EntF-Push or ```--recipe pull``` corresponding to EntF-Pull.

 ## Citation

```bibtex
@inproceedings{wen2023is,
    title={Is Adversarial Training Really a Silver Bullet for Mitigating Data Poisoning?},
    author={Rui Wen and Zhengyu Zhao and Zhuoran Liu and Michael Backes and Tianhao Wang and Yang Zhang},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=zKvm1ETDOq}
}
```

## Contact

If you are interested in our work, feel free to drop [me](https://wenruiustc.github.io/) an email at rui.wen@cispa.de


## Acknowledgement

We would like to acknowledge the work of [Fowl et al.](https://github.com/lhfowl/adversarial_poisons) for their excellent framework for generating poisons based on adversarial examples. Our code leverages their code for the gradient descent and poison saving parts.