# Privacy-Preserving Retrieval for Auditable Clinical Language Modeling on Real-World Radiology Data 

kNN-LM • RETOMATON • FAISS Datastores • Instruction-Tuned Models

Consolidated Repository of Experiments for Retrieval-based Language Modelling using the Retomaton &amp; KNNLM Models. Dummy radiology notes provided as an example derived from the MIMIC IV Dataset [1], datasets can be downloaded from Physionet (URL still under review by Physionet [2]):

Purswani, N., Schlegel, V. & Bharath, A. A. Privacy-preserving retrieval for auditable clinical language modeling on real-world radiology data (version 1.0). PhysioNet Dataset (2026). https://doi.org/10.13026/***** (Full URL to be updated upon dataset approval, currently under review by Physionet authors)

The original code for retrieval-based language models is here, we have adapted it from Khandelwal, Alon et al.[3-5]: https://github.com/neulab/knn-transformers 

This repository contains all scripts, environments, and utilities required to:
- Build FAISS-based datastores from clinical radiology notes  
- Run kNN-LM and RETOMATON for domain adaptation of models with next-token retrieval evaluation  
- Fine-tune instruction-tuned models on clinical radiology notes
- Extract hidden states for large-scale datastore construction  

The workflow is optimized for **clinical NLP**, **privacy-preserving modeling**, and **non-parametric adaptation** using external memory instead of weight updates. The authors have submitted a paper for review and will make the pre-print available through github once it's uploaded. 

[1] Johnson, A. E. W. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci. Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x 

[2] Goldberger, A. L. et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101, e215–e220 (2000). https://doi.org/10.1161/01.cir.101.23.e215

[3] Khandelwal, U., et al. Generalization through memorization: nearest neighbor language models. In International Conference on Learning Representations (ICLR) (2020). https://doi.org/10.48550/arXiv.1911.00172

[4] Khandelwal, U., et al. Nearest neighbor machine translation. In International Conference on Learning Representations (ICLR) (2021). https://doi.org/10.48550/arXiv.2010.00710

[5] Alon, U. et al. Neuro-symbolic language modeling with automaton-augmented retrieval. In ICML Workshop on Knowledge-Driven Representation Learning for Machine Learning (KRLM) (2022). https://doi.org/10.48550/arXiv.2201.12431

---

## 0. Pre-requisite: Dataset Downloads from Physionet (URL to be Updated upon approval)

Log into Physionet and download datasets inside the Dummy_Data_Github folder:

<img width="693" height="619" alt="image" src="https://github.com/user-attachments/assets/fbb6c7fb-65dc-43c6-9fbb-961ee81e81c0" />

Ready to use files with applied chat template are here (they can be used for retrieval and finetuning): 

mimic_inspired_test_context_impression-finetune.jsonl   
mimic_inspired_val_context_impression-finetune.jsonl
mimic_inspired_train_context_impression-finetune.jsonl

To run these on google colab ->

Copy and paste them into the folder: /retomaton_healthcare_paper/colab_workflow/execution_scripts/mock_datasets_radiology_jsonl/

To run these on an HPC cluster ->

Copy and paste the files below into folder: /retomaton_healthcare_paper/hpc_workflow/dummy_datasets/

(The .csv files are needed to apply the chat template prior)
mimic_inspired_test_context_impression-finetune.jsonl   
mimic_inspired_val_context_impression-finetune.jsonl
mimic_inspired_train_context_impression-finetune.jsonl
mimic_inspired_test_context_impression.csv  
mimic_inspired_val_context_impression.csv
mimic_inspired_train_context_impression.csv

To re-create the full index from the paper (use HPC cluster with at least 1 40 GB NVDIA GPU, and 150 GB RAM):  -->

Copy and paste the files below (obtained from physionet ./Index_Creation_Files/subsampled_datasets/) into this folder: /retomaton_healthcare_paper/hpc_workflow/dummy_datasets/

test_10p.csv
train_10p.csv
val_10p.csv

## 1. Core Scripts (colab workflow) ./colab_workflow/core_scripts: 

The ./core_scripts folder doesn't need to be modified. Important files include:

retomaton.py  (Retomaton wrapper from Neubig Lab)

knnlm.py (Knnlm wrapper, which retomaton depends on)

run_clm_chat.py (Modified template for causal language modelling with retomaton and knnlm, using the Hugging Face template)

4_generations_perplexity_debug3.py : Runs text generations for base models, retomaton and finetuned models and computex perplexity scores.

## 2. Execution Scripts (colab workflow) ./colab_workflow/execution_scripts:

./mock_datasets_radiology_jsonl/ folder: should contain files accessed in step 0. from Physionet in the correct prompt template and format. Specifically, need files: 

mimic_inspired_test_context_impression-finetune.jsonl   
mimic_inspired_val_context_impression-finetune.jsonl
mimic_inspired_train_context_impression-finetune.jsonl

1_run_knn_saver.ipynb : Script that calls run_clm_chat.py and the retomaton.py and knnlm.py wrappers to generate a datastore. 
2_build_faiss_index.ipynb : Script that builds faiss index for retrieval based language modelling using the datastore built in step 1.
3_run_generations.ipynb : runs text generation with the retrieval-based LM approach and the externally created FAISS Datastore. 

----

## 3. Environment Setup for HPC Cluster Workflow

The `env_config/` folder provides **three Conda environments**, each dedicated to a specific stage of the workflow.

1. neubig_benchmarks_environment.yml (for evaluation)
2. neubig_finetune_environment.yml (for supervised finetuning)
3. neubig_instruct_environment.yml (for next-token retrieval)


## 4. hpc_workflow: Core Scripts

The ./core_scripts folder doesn't need to be modified. Important files include:

retomaton.py  (Retomaton wrapper from Neubig Lab)

knnlm.py (Knnlm wrapper, which retomaton depends on)

run_clm_chat.py (Modified template for causal language modelling with retomaton and knnlm, using the Hugging Face template)

4_generations_perplexity_debug3.py : Runs text generations for base models, retomaton and finetuned models and computex perplexity scores.

5_all_metrics.py : Evaluation script comparing finetuning, next-token retrieval for different parameter combinations against base models (not finetuned or enhanced with knnlm and retomaton) with multiple benchmarks (reference and reference free).


## 5. hpc_workflow: Bash Files

run steps 0-5.bash in sequence

0_jsonl_data_prompt_format.bash : Process dummy input data into jsonl format for instruction tuned prompting and finetuning 

1_save_knn_datastore.bash : Save the datastore containing previous/next token sequences (expected size for dummy dataset stored in dummy_datasets of ~280K Key/Val pairs)

2_build_index_base.bash : Build a faiss index from the datastores created in step 1. for next-token retrieval

3_run_sft.bash : Run supervised finetuning

4_run_generations_debug.bash : Generate outputs for retomaton, base models and finetuned models from previous steps

5_run_all_metrics.bash : Evaluate models based on ROUGE-L scores, Perplexity, Hallucination Metrics etc. 

## 6. Citations & Accreditations: 

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

Pre-print and citation Link for this repository will be updated soon with PhysioNet URL. Please let us know if you have any suggestions for improvement. 
