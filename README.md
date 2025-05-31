# TutorMind at BEA 2025 Shared Task: 

This repository contains the code and data used in our submission to the BEA 2025 Shared Task on Pedagogical Ability Assessment. Our work focuses on the **Mistake Identification** dimension, evaluating whether AI tutor responses correctly identify student mistakes in educational dialogues.

## Structure

- `data/`  
  Contains the data used in our code. This includes:
  - *Mistake_identification_train_split_chat_format.jsonl:* the training subset (80%)
  - *Mistake_identification_validation_split_chat_format.jsonl:* the validation subset (20%)
  - *Mistake_identification_augm_train_split_chat_format.jsonl:* the training subset (80%) + data augmentation
  - *augmented_full_devset.json:* original full devset + data augmentation

- `data-augmentation-command-R+/`  
  Script and prompt for generating synthetic examples using Command R+, targeting underrepresented labels (`No` and `To some extent`).

- `gpt-4o-mini-experiments/`  
  Fine-tuning and inference code for GPT-4o-mini using OpenAI’s SFT pipeline.

- `mistral-7b-instruct-experiments/`  
  Fine-tuning code and evaluation for Mistral-7B using Unsloth with 4-bit quantization and LoRA adapters.

- `LLaMA-3.1-8B-instruct-experiments/`  
  Scripts for training and evaluating LLaMA-3.1-8B on the Mistake Identification task.

## Highlights

- GPT-4o-mini fine-tuned on the full devset achieved **71.63% strict macro-F1** on the official blind test set, ranking **second** in the shared task.
- Introduced targeted data augmentation using Command R+ to improve minority class performance.
- Benchmarked multiple LLMs (GPT-4o-mini, Mistral-7B, LLaMA-3.1-8B) under strict and lenient evaluation settings.

## Citations

**BEA 2025 Shared Task**  
Ekaterina Kochmar, Kaushal Kumar Maurya, Kseniia Petukhova, KV Aditya Srivatsa, Anaïs Tack, and Justin Vasselli.  
*Findings of the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-Powered Tutors.*  
In Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications.

**MRBench Dataset**  
Kaushal Kumar Maurya, KV Aditya Srivatsa, Kseniia Petukhova, and Ekaterina Kochmar.  
*Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors.*  
In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1234–1251.


