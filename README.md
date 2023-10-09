# dl4nlp-text-summarization



Project structure:
- experiments: 
How to combate hallucination; with prompt engineering or Chain of Verification (CoVe). 
- jobs: files used for job queuing on server using SLURM.  
- metric: contain hallucationa metric such as factsumm and summac. Download bleurt model within this folder, see notebook on how to install and bug fix.
- model

training--files

- Lots of bugs and package to do manually, see hallucination.ipynb
  - For factsumm to get the package dir. :
      - python in terminal
      - >>> import factsumm
      - >>> factsumm.__file__

- ??conda env create -f install_env.yml??

- Run: 'python train.py --wandb-mode disabled

- Lots of bugs and package to do manually, see hallucination.ipynb
