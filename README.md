# IRanker: Towards Ranking Foundation Model

<p align="center">
    <a href="https://ulab-uiuc.github.io/GraphRouter/">
        <img alt="Build" src="https://img.shields.io/badge/Project-Page-blue">
    </a>
    <a href="http://arxiv.org/abs/2410.03834">
        <img alt="Build" src="https://img.shields.io/badge/arXiv-2410.11001-red?logo=arxiv">
    </a>
    <!-- <a href="xxx">
        <img alt="Build" src="https://img.shields.io/badge/Twitter-black?logo=X">
    </a> -->
    <a href="https://github.com/ulab-uiuc/GraphRouter/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <br>
    <a href="https://github.com/ulab-uiuc/GraphRouter">
        <img alt="Build" src="https://img.shields.io/github/stars/ulab-uiuc/GraphRouter">
    </a>
    <a href="https://github.com/ulab-uiuc/GraphRouter">
        <img alt="Build" src="https://img.shields.io/github/forks/ulab-uiuc/GraphRouter">
    </a>
    <a href="https://github.com/ulab-uiuc/GraphRouter">
        <img alt="Build" src="https://img.shields.io/github/issues/ulab-uiuc/GraphRouter">
    </a>
</p>


<p align="center">
    <a href="https://ulab-uiuc.github.io/GraphRouter/">🌐 Project Page</a> |
    <a href="http://arxiv.org/abs/2410.03834">📜 arXiv</a>
    <!-- <a href="xxx">📮 Twitter Post</a> -->
<p>


<!-- ![Method](./figures/model.png) -->

<div align="center">
  <img src="./figures/model.png" width="700" alt="GoR">
</div>


## 📌Preliminary


### Environment Setup

```shell
conda create -n iranker python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib

```
## 📊 Dataset Preparation

This section outlines the steps to generate the datasets used for DRanker and IRanker training and evaluation.

### DRanker Dataset

To generate the DRanker dataset, run the following command:

```bash
python examples/data_preprocess/direct_data_generation.py
```

The processed dataset will be saved to: `data/direct_ranking`

### IRanker Dataset

To generate the IRanker dataset, execute this script:

```bash
python examples/data_preprocess/iterative_data_generation.py
```

The processed dataset will be saved to: `data/iterative_ranking`

### Raw Dataset

The original raw dataset is available for download from Hugging Face:

**Dataset Repository:** [ulab-ai/Ranking-bench](https://huggingface.co/datasets/ulab-ai/Ranking-bench)


## ⭐Experiments


### Training and Evaluation

Run experiments and print/save evaluation results on metrics Performance, Cost, and Reward. You can edit the hyperparameters in `configs/config.yaml` or using your own config_file.


```bash
python run_exp.py --config_file [config]
```

## 🔍 Evaluation

### Running Evaluation

To evaluate a model on a specific dataset, use the following command:

```bash
python eval/eval.py --dataset <dataset_name> --model_path <path_to_model>
```

### Parameters

- `--dataset`: Specifies the dataset to evaluate on
- `--model_path`: Path to the trained model you want to evaluate

### Supported Datasets

The evaluation script supports the following datasets:

#### Passage Ranking
- `Passage-5` 
- `Passage-7` 
- `Passage-9` 

#### Router Tasks
- `Router-Performance`
- `Router-Balance` 
- `Router-Cost` 

#### Recommendation Systems
- `Rec-Movie` 
- `Rec-Music` 
- `Rec-Game`


## Citation

```bibtex
@inproceedings{feng2024graphrouter,
  title={Graphrouter: A graph-based router for llm selections},
  author={Feng, Tao and Shen, Yanzhen and You, Jiaxuan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}
```


<!-- <picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ulab-uiuc%2FGraphEval&theme=dark&type=Date">
<img width="100%" src="https://api.star-history.com/svg?repos=ulab-uiuc%2FGraphEval&type=Date">
</picture> -->
