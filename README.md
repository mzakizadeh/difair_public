# DiFair Benchmark: EMNLP 2023

This repository houses the evaluation suite and the generated dataset variant utilized in our publication for EMNLP 2023 titled ["DiFair: A Benchmark for Disentangled Assessment of Gender Knowledge and Bias"](https://aclanthology.org/2023.findings-emnlp.127/). The aim is to provide the community with the resources to replicate our findings and further extend upon the established benchmark.

## Overview

DiFair serves as a meticulous endeavor to address the oversight in evaluating the impact of bias mitigation on useful gender knowledge while assessing gender neutrality in pretrained language models. Our manually curated dataset, hinging on masked language modeling objectives, introduces a unified metric, the Gender Invariance Score (GIS). This metric delves into not only quantifying a model's biased tendencies but also assessing the preservation of useful gender knowledge. Through DiFair, we benchmark several widely-regarded pretrained models and debiasing techniques, and our empirical findings echo the existing narrative on gender biases and the trade-off entailed in debiasing efforts. The repository provides a preprocessed version of the original data, sans special tokens and detailed labels, alongside the evaluation code to uphold ethical considerations.

## Dataset

In alignment with our commitment to ethical research practices, we have opted not to publicly release the original annotated dataset or the pipeline used to generate variations of the dataset. This is primarily due to the synthetic nature of the DiFair dataset, which could potentially be misused beyond research purposes. To support the integrity of our research and enable verification of our results, we have made available a version of the dataset that was employed in the computation of the results presented in Tables 1 and 3 of the paper. This dataset can be accessed via [this link](https://drive.google.com/file/d/122PmTa5KBde5zbsTisGJLHhjZfcbWXWU/view?usp=sharing).

For researchers interested in accessing the original dataset, we can make it available upon request. Please reach out to us via email to gain access.

## Environment Setup

Ensure your environment is properly configured to run the evaluation code by following these steps:

- Create and activate the conda environment:
```bash
conda env create --name difair --file=environment.yml
conda activate difair
```

## Evaluation

Utilize the evaluation suite to compute the Gender Invariance Score (GIS) for an arbitrary model as follows:

1. Execute the `evaluate.py` script:
```bash
python evaluate.py --model [PATH_TO_MODEL] [OTHER_OPTIONS]
```

2. To run the primary experiment, use the provided Makefile:
```bash
make main_experiment > outputs/main_experiment.log
```

## Parameter Configuration

Configure the evaluation script using the parameters listed below:

### `evaluate.py`

| Parameter Name          | Description                                             | Default Value                    |
|-------------------------|---------------------------------------------------------|----------------------------------|
| `--dataset_path`        | Path to the input .csv dataset                          | `generated/difair_-30y_now.csv`  |
| `--model_name` (Required) | Name of the Hugging Face pretrained model               |                                  |
| `--model_path`          | Path to the Hugging Face pretrained model               |                                  |
| `--adapter`             | Path to the pretrained adapter modules                  |                                  |
| `--masculine_words_path`| Path to text file containing masculine words            | `data/masculine_words.txt`       |
| `--feminine_words_path` | Path to text file containing feminine words             | `data/feminine_words.txt`        |
| `--no_balance_dataset`  | Toggle to balance/unbalance the dataset for tests       | N/A                              |
| `--no_normalization`    | Toggle to normalize/unnormalize probability distribution| N/A                              |

## Contact Information

For any inquiries regarding the DiFair Benchmark, dataset access, or related research questions, please feel free to contact me:
- Name: Mahdi Zakizadeh
- Email: m.zakizadeh@khatam.ac.ir

## Citation

```
@inproceedings{zakizadeh-etal-2023-difair,
    title = "DiFair: A Benchmark for Disentangled Assessment of Gender Knowledge and Bias",
    author = "Zakizadeh, Mahdi  and
      Miandoab, Kaveh  and
      Pilehvar, Mohammad",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.127",
    doi = "10.18653/v1/2023.findings-emnlp.127",
    pages = "1897--1914",
}
```
