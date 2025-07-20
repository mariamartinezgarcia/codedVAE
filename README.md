# Coded VAE

Official implementation of the paper [**Improved Variational Inference in Discrete VAEs using Error Correcting Codes**](https://openreview.net/pdf?id=66P2ioAYY4), **accepted at Uncertainty in Artificial Intelligence 2025 (UAI 2025)**.

## Overview

This work introduces a novel method to improve inference in discrete Variational Autoencoders (VAEs) by reframing the inference problem through a generative perspective. **We conceptualize the model as a communication system and leverage Error Correcting Codes (ECCs) to introduce redundancy in latent representations**, allowing the variational posterior to produce more accurate estimates and reduce the variational gap.

We present a proof-of-concept using a Discrete VAE with binary latent variables and low-complexity repetition codes, and extend it to a hierarchical structure for disentangling global and local data features. Our approach significantly improves generation quality, data reconstruction, and uncertainty calibration.

---

## Repository Structure

This repository contains the source code and demonstration scripts for training and evaluating the following model variants:
- **Uncoded model**
- **Coded model with repetition codes**
- **Hierarchical coded model with polar codes**
- **Hierarchical coded model without polar codes**

#### `/src` - Coded VAE Source Code

This folder contains the source code of the model (includes the implementation for all the configurations). 


#### `/codes` - Repetition Codes

This folder contains `.pkl` files with the matrices for encoding and decoding for each repetition code. The filename indicates the code rate.

**Example**:  
`rep_matrices_5_50.pkl` → repetition code with code rate 5/50.

**Custom code**. To generate matrices for a custom code rate, use:

```bash
python script_repetition_codes.py -m <num_information_bits> -c <num_code_bits>
```

Arguments:
- `-m` : Number of information bits (dimension of `m`)
- `-c` : Number of code bits (dimension of `c`)

#### `/checkpoints` - Model Checkpoints

This filder contains checkpoints for the following Coded VAE configurations, all trained on FMNIST: 
- Uncoded model with 8 latent bits (`UNCODED_fmnist_8_enc_dcgan_dec_cnnskip_beta15.pt`)
- Coded model with an 8/240 repetition code (`CODED_rep_fmnist_8_240_equal_enc_dcgan_dec_cnnskip_beta15.pt`)
- Hierarchical model with symmetrical branches using a 5/100 repetition code and polar code (`HIER_fmnist_5_100_equal_enc_dcgan_dec_cnnskip_beta15_polar.pt`)

#### `/configs` - Configuration Files

This folder contains the configuration files for the different demo scripts provided.

---

## Checkpoints

### Model (trained on FMNIST)

Included in the `/checkpoints` folder:
- Uncoded model with 8 latent bits (`UNCODED_fmnist_8_enc_dcgan_dec_cnnskip_beta15.pt`)
- Coded model with an 8/240 repetition code (`CODED_rep_fmnist_8_240_equal_enc_dcgan_dec_cnnskip_beta15.pt`)
- Hierarchical model with symmetrical branches using a 5/100 repetition code and polar code (`HIER_fmnist_5_100_equal_enc_dcgan_dec_cnnskip_beta15_polar.pt`)

### Classifiers

We also provide the checkpoints of the classifiers used for automatic evaluation and computing semantic accuracy. 

- `classifier_network_fmnist.pt`
- `classifier_network_mnist.pt`  


---

## Training

We provide demo scripts to show how to train uncoded models (`train_uncoded.py`), coded models with repetition codes (`train_coded.py`) and coded models with hierarchical architecture (`train_coded_hierarchical.py`). In all the cases, the only argument required is:

- `-c`: configuration file.

**Example:**
```bash
python train_coded.py -c config_train_coded.py
```

Configuration files must be stored in the `/configs` folder. 

---

## Evaluation

We provide demo scripts to show how to evaluate uncoded models (`evaluation_uncoded.py`), coded models with repetition codes (`evaluation_coded.py`) and coded models with hierarchical architecture (`evaluation_coded_hierarchical.py`).  In all the cases, the only argument required is:

- `-c`: configuration file.


**Example:**
```bash
python evaluation_coded.py -c config_eval_coded.py
```

Configuration files must be stored in the `/configs` folder.

---

## Hierarchical Models

The implementation supports various hierarchical configurations:

- **Number of branches**:  
  You can define multiple branches by specifying the number of information and coded bits in (ascending) order in the configuration file.

  **Example**: Hierarchical code with 3 branches. We use a 5/100 code for the first branch, and a 5/50 for the second and third branches
  ```
  bits_info: [5, 5, 5]
  bits_code: [100, 50, 50]
  ```

- **Polar codes**:  
  Enable or disable the polar code in the hierarchical model by setting the `polar` flag to `True` or `False` in the configuration file.

---

## Contact

For questions or issues, please contact María Martínez-García via the email martinez-garcia@cs.uni-saarland.de .

---