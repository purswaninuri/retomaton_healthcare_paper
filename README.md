# 🔍 Next-Token Retrieval for Clinical Language Modeling  
kNN-LM • RETOMATON • FAISS Datastores • Instruction-Tuned Models

Consolidated Repository of Experiments for Next-Token Retrieval using the Retomaton &amp; KNNLM Models. Synthetic radiology notes provided as an example, generated using ChatGPT 5.2

The original code for next token retrieval is here, we have adapted it: https://github.com/neulab/knn-transformers 

This repository contains all scripts, environments, and utilities required to:
- build FAISS-based datastores from clinical notes  
- run kNN-LM and RETOMATON for domain adaptation of models with next-token retrieval evaluation  
- fine-tune instruction-tuned models  
- extract hidden states for large-scale datastore construction  

The workflow is optimized for **clinical NLP**, **privacy-preserving modeling**, and **non-parametric adaptation** using external memory instead of weight updates.



---

# 📦 1. Environment Setup

The `env_config/` folder provides **three Conda environments**, each dedicated to a specific stage of the workflow.

1. neubig_benchmarks_environment.yml (for evaluation)
2. neubig_finetune_environment.yml (for supervised finetuning)
3. neubig_instruct_environment.yml (for next-token retrieval)


# 2. hpc_workflow: Core Scripts

The ./core_scripts folder doesn't need to be modified. Important files include:

retomaton.py  (Retomaton wrapper from Neubig Lab)

knnlm.py (Knnlm wrapper, which retomaton depends on)

run_clm_chat.py (Modified template for causal language modelling with retomaton and knnlm, using the Hugging Face template)

4_generations_perplexity_debug.py : Runs text generations for base models, retomaton and finetuned models and computex perplexity scores.

5_all_metrics.py : Evaluation script comparing finetuning, next-token retrieval for different parameter combinations against base models (not finetuned or enhanced with knnlm and retomaton) with multiple benchmarks (reference and reference free).


# 3. hpc_workflow: Bash Files

run steps 0-5.bash in sequence

0_jsonl_data_prompt_format.bash : Process dummy input data into jsonl format for instruction tuned prompting and finetuning 

1_save_knn_datastore.bash : Save the datastore containing previous/next token sequences (expected size for dummy dataset stored in dummy_datasets of ~280K Key/Val pairs)

2_build_index_base.bash : Build a faiss index from the datastores created in step 1. for next-token retrieval

3_run_sft.bash : Run supervised finetuning

4_run_generations_debug.bash : Generate outputs for retomaton, base models and finetuned models from previous steps

5_run_all_metrics.bash : Evaluate models based on ROUGE-L scores, Perplexity, Hallucination Metrics etc. 


# 4. Colab files folder:

Version of the workflow to run on google collab pro+ if environment setup on HPC cluster is difficult. The notebooks use high RAM and 1 L40 NVIDIA-GPU (paid version). The colab version of the code uses FAISS-CPU as FAISS-GPU has set-up problems. 

To run this part, just download the ./colab_files subfolder and upload it to your local google drive. Then, with your file paths it can be ran through colab high RAM + GPU notebooks. Connection to Google Drive must be enabled. 

# 5. Citations & Accreditations: 

This code was based on prior work from Alon et al. and Khandelwal et al. 

Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval: https://arxiv.org/abs/2201.12431

@inproceedings{alon2022neuro,
  title     = {Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval},
  author    = {Alon, Uri and Xu, Frank and He, Junxian and Sengupta, Sudipta and Roth, Dan and Neubig, Graham},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  pages     = {468--485},
  year      = {2022},
  publisher = {PMLR}
}

Generalization through Memorization: Nearest Neighbor Language Models: https://arxiv.org/abs/1911.00172

@inproceedings{khandelwal2020generalization,
  title     = {Generalization through Memorization: Nearest Neighbor Language Models},
  author    = {Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}

If you find this repository useful, please consider starring ⭐ it and citing the papers above. 

Preprint/Citation Link for my article to be updated soon. 
