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



checkpoints = {

    # t5-small : Paulius
    "t5-s": "t5-small",
    "t5-s-xsum": "pki/t5-small-finetuned_xsum",   
    
    # t5-large : Luka
    "t5-large": "t5-large",
    "t5-large-xsum": "sysresearch101/t5-large-finetuned-xsum",

    # t5-large xsum-cnn Skipping for now
    # "t5-large-xsum-cnn" : "sysresearch101/t5-large-finetuned-xsum-cnn",

    # google/pegasus-xsum" :  Erencan
    "pegasus-xsum": "google/pegasus-xsum",
}