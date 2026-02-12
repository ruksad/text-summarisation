# Abstractive Text Summarisation

An applied NLP project exploring state-of-the-art abstractive summarisation using encoder–decoder transformers (T5 and BART) with and without PEFT (Parameter-Efficient Fine-Tuning). The repository contains well-documented notebooks that train, evaluate, and compare baselines, PEFT approaches, beam configurations, and hallucination-aware decoding.


## Highlights
- Models: T5 (Small, Base) and BART (Base)
- Training strategies: full fine-tuning and PEFT (LoRA/Adapters)
- Decoding: beam search, nucleus/top-k sampling, and hallucination-mitigation ("HalluTree")
- Comparative experiments across configurations with reproducible notebooks
- Ready-to-run on CPU/GPU; minimal setup (Hugging Face + PyTorch)


## Repository Structure
- src/
  - Final_T5_Small_PEFT.ipynb — PEFT fine-tuning of T5-Small
  - Final_T5_Base_PEFT_HalluTree.ipynb — T5-Base with PEFT and hallucination-aware decoding
  - Final_Bart_Base_PEFT.ipynb — PEFT fine-tuning of BART-Base
  - T5_Base_No_PEFT_N-Beams.ipynb — T5-Base baseline with beam-search variations
  - T5_Small_No_PEFT.ipynb — T5-Small baseline
  - Extractive_Abstractive_Summary.ipynb — hybrid approach: extractive pre-step with abstractive refinement
- Tensor Troopers Project Report.pdf — project report with methodology and results


## Quick Start
1) Create an environment
- Python >= 3.9 recommended
- Create a virtual environment (examples for Conda and venv):

```bash
# Conda
conda create -n summarisation python=3.10 -y
conda activate summarisation

# or venv
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA/CPU
pip install transformers datasets peft accelerate evaluate rouge-score nltk sentencepiece
jupyterlab ipywidgets matplotlib seaborn tqdm
```

Notes:
- Choose the proper PyTorch build for your system (CUDA/CPU). See https://pytorch.org/get-started/locally/
- If running in notebooks, ensure ipywidgets is enabled.

3) Open the notebooks
- Start JupyterLab or Notebook and open files under src/:

```bash
jupyter lab  # or: jupyter notebook
```


## Datasets
The notebooks are written to work seamlessly with Hugging Face Datasets. Common choices include:
- CNN/DailyMail
- XSum
- Gigaword

You can switch datasets by editing the dataset loading cell (datasets.load_dataset) and adjusting preprocessing (input/target fields, max lengths).


## Models and Methods
- Baselines:
  - T5-Small and T5-Base without PEFT
  - Beam search experiments with N-beams and length/coverage settings
- PEFT Variants:
  - LoRA/Adapters via the peft library for parameter-efficient fine-tuning
  - Applied to T5-Small, T5-Base, and BART-Base
- Hallucination Mitigation (HalluTree):
  - A decoding-time strategy that prunes candidates using factuality/consistency heuristics
  - Implemented in Final_T5_Base_PEFT_HalluTree.ipynb as an experimental method
- Hybrid (Extractive + Abstractive):
  - Preliminary extractive selection (e.g., sentence ranking) followed by a transformer-based abstractive step


## Reproducing Key Experiments
Each notebook contains cells to configure hyperparameters and training/evaluation loops. Typical workflow:
1) Select dataset and subset/split
2) Tokenise with the corresponding model tokenizer
3) Configure training args (batch size, epochs, learning rate, gradient accumulation, max source/target lengths)
4) Choose decoding strategy (beam size, length penalty, top-k/p)
5) Train (either full FT or PEFT) and evaluate with ROUGE

Example hyperparameters (guidance; tune per dataset/hardware):
- Sequence lengths: max_source_length=512, max_target_length=128
- Optimiser: AdamW, lr=5e-5 to 2e-4; weight decay=0.01
- Scheduler: linear with warmup (e.g., 500 steps)
- Batch sizes: 8–32 (use gradient accumulation to fit memory)
- Beams: 1 (greedy) to 8; try 4–6 for a quality/latency balance


## Evaluation
- Automatic metrics: ROUGE-1/2/L via evaluate + rouge-score
- Optional: BLEU, METEOR
- Qualitative checks: faithfulness/consistency, redundancy, readability
- For HalluTree runs, compare factuality using available checks/heuristics described in the notebook


## Results Snapshot
- PEFT typically matches or surpasses full FT at a fraction of trainable parameters
- T5-Base outperforms T5-Small but requires more compute; BART-Base is a strong alternative
- Hallucination-aware decoding can reduce unsupported claims at a small cost to speed

Refer to Tensor Troopers Project Report.pdf for comprehensive tables and discussion.


## Tips for Best Performance
- Enable gradient checkpointing to reduce memory usage on larger models
- Mixed-precision (fp16/bf16) with accelerate speeds up training on modern GPUs
- Use accelerate config to tailor device/DistributedDataParallel settings
- Save and load PEFT adapters to quickly switch between base checkpoints


## Troubleshooting
- CUDA out of memory: lower batch size, shorten sequences, increase gradient accumulation, or use smaller model
- Tokenizer/model mismatch: ensure the checkpoint name matches the tokenizer
- Low ROUGE: review preprocessing (truncation, special tokens), try longer target lengths or higher beams


## Citation
If you use this work in your research or projects, please cite the base libraries and models:

- T5: Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- BART: Lewis et al., BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation
- PEFT: Mangrulkar et al., PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning


## License
This project is released for educational and research purposes. Adjust or add a formal license file (e.g., MIT/Apache-2.0) as needed for your use case.


## Acknowledgements
- Hugging Face Transformers, Datasets, Evaluate, and PEFT
- PyTorch
- Open-source contributors and prior summarisation research
