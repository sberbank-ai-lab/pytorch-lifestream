# Setup using conda

```sh
# download Miniconda from https://conda.io/
curl -OL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# execute intall file
bash Miniconda*.sh
# relogin to apply PATH changes
exit

# install pytorch
# instead of 10.1 use a current version of cuda, see nvidia-smi
conda install pytorch cudatoolkit=10.1 -c pytorch
# check CUDA version
python -c 'import torch; print(torch.version.cuda)'
# chech torch GPU
# See question: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
python -c 'import torch; print(torch.rand(2,3).cuda())'

# clone repo
git clone 

# install dependencies
conda install -c conda-forge pyspark 
pip install -r requirements.txt
```

# Setup and test using pipenv

```sh
# Ubuntu 18.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv install --dev
pipenv install "git+ssh://git@github.com/sberbank-ai-lab/embeddings-valid.git#ref=0.0.1&egg=embeddings-validation"

pipenv shell

pytest
```

# Embeddings validation

```
# install membeddings_validation tool
git clone git@github.com:sberbank-ai-lab/embeddings-valid.git
cd embeddings-valid
python setup.py install

# run luigi server
luigid

# check embedding validation progress at `http://localhost:8082/`
```

# Run scenario

```sh
cd experiments

# See README.md from the specific project
```

# Final results
```

LightGBM:                                     mean t_int_l t_int_h    std                           values
    Designed features:                                                                                    
        Age group   |   0.626 \pm 0.004   | 0.6323  0.6268  0.6379 0.0040  [0.628 0.630 0.630 0.635 0.638]
        Gender      |   0.875 \pm 0.004   | 0.8798  0.8751  0.8844 0.0033  [0.874 0.879 0.881 0.882 0.882]
        Assessment  |   0.591 \pm 0.003   | 0.5909  0.5872  0.5946 0.0027  [0.588 0.588 0.591 0.593 0.594]
        Retail      |   0.545 \pm 0.001   | 0.5454  0.5439  0.5469 0.0011  [0.544 0.545 0.545 0.546 0.547]
    CPC embeddings:         
        Age group   |   0.595 \pm 0.004   | 0.5939  0.5900  0.5977 0.0028  [0.590 0.593 0.594 0.595 0.598]
        Gender      |   0.848 \pm 0.004   | 0.8572  0.8545  0.8600 0.0020  [0.855 0.856 0.857 0.858 0.860]
        Assessment  |   0.584 \pm 0.004   | 0.5838  0.5790  0.5886 0.0035  [0.578 0.584 0.585 0.586 0.587]
        Retail      |   0.527 \pm 0.001   | 0.5265  0.5252  0.5278 0.0009  [0.525 0.526 0.527 0.527 0.528]
    MeLES embeddings:       
        Age group   |   0.639 \pm 0.006   | 0.6419  0.6372  0.6466 0.0034  [0.636 0.642 0.643 0.644 0.645]
        Gender      |   0.872 \pm 0.005   | 0.8821  0.8801  0.8840 0.0014  [0.880 0.882 0.882 0.882 0.884]
        Assessment  |   0.604 \pm 0.003   | 0.6041  0.5994  0.6088 0.0034  [0.598 0.604 0.605 0.606 0.607]
        Retail      |   0.544 \pm 0.001   | 0.5439  0.5421  0.5457 0.0013  [0.542 0.544 0.544 0.545 0.545]

Scores:                       
    Supervised learning:    
        Age group   |   0.631 \pm 0.010   | 0.6285  0.6172  0.6399 0.0082  [0.619 0.625 0.626 0.631 0.641]
        Gender      |   0.871 \pm 0.007   | 0.8741  0.8654  0.8828 0.0063  [0.865 0.872 0.873 0.879 0.881]
        Assessment  |   0.601 \pm 0.006   | 0.6010  0.5948  0.6072 0.0045  [0.596 0.600 0.600 0.601 0.608]
        Retail      |   0.543 \pm 0.002   | 0.5425  0.5403  0.5447 0.0016  [0.540 0.542 0.542 0.544 0.544]
    CPC fine-tuning:        
        Age group   |   0.621 \pm 0.007   | 0.6210  0.6166  0.6254 0.0032  [0.617 0.618 0.622 0.623 0.625]
        Gender      |   0.873 \pm 0.007   | 0.8777  0.8726  0.8829 0.0037  [0.874 0.875 0.878 0.879 0.883]
        Assessment  |   0.611 \pm 0.005   | 0.6115  0.6047  0.6184 0.0049  [0.603 0.611 0.613 0.615 0.616]
        Retail      |   0.546 \pm 0.003   | 0.5461  0.5429  0.5492 0.0023  [0.542 0.546 0.546 0.547 0.548]
    MeLES fine-tuning:      
        Age group   |   0.643 \pm 0.007   | 0.6383  0.6331  0.6435 0.0037  [0.632 0.637 0.641 0.641 0.641]
        Gender      |   0.888 \pm 0.002   | 0.8959  0.8910  0.9009 0.0036  [0.890 0.896 0.898 0.898 0.898]
        Assessment  |   0.614 \pm 0.003   | 0.6135  0.6095  0.6176 0.0029  [0.608 0.614 0.615 0.615 0.616]
        Retail      |   0.549 \pm 0.001   | 0.5490  0.5479  0.5500 0.0008  [0.548 0.549 0.549 0.550 0.550]



LightGBM:                                     mean t_int_l t_int_h    std                           values
    Designed features:                                                                                    
        Age group   |   0.626 \pm 0.004   | 0.6323  0.6268  0.6379 0.0040  [0.628 0.630 0.630 0.635 0.638] | 0.632 \pm 0.005 |  0.0% \pm  0.3%
        Gender      |   0.875 \pm 0.004   | 0.8798  0.8751  0.8844 0.0033  [0.874 0.879 0.881 0.882 0.882] | 0.880 \pm 0.004 |  0.0% \pm  0.2%
        Assessment  |   0.591 \pm 0.003   | 0.5909  0.5872  0.5946 0.0027  [0.588 0.588 0.591 0.593 0.594] | 0.591 \pm 0.003 |  0.0% \pm  0.3%
        Retail      |   0.545 \pm 0.001   | 0.5454  0.5439  0.5469 0.0011  [0.544 0.545 0.545 0.546 0.547] | 0.545 \pm 0.001 |  0.0% \pm  0.1%
    CPC embeddings:         
        Age group   |   0.595 \pm 0.004   | 0.5939  0.5900  0.5977 0.0028  [0.590 0.593 0.594 0.595 0.598] | 0.594 \pm 0.004 | -6.0% \pm  0.3%
        Gender      |   0.848 \pm 0.004   | 0.8572  0.8545  0.8600 0.0020  [0.855 0.856 0.857 0.858 0.860] | 0.857 \pm 0.002 | -2.5% \pm  0.2%
        Assessment  |   0.584 \pm 0.004   | 0.5838  0.5790  0.5886 0.0035  [0.578 0.584 0.585 0.586 0.587] | 0.584 \pm 0.004 | -1.2% \pm  0.3%
        Retail      |   0.527 \pm 0.001   | 0.5265  0.5252  0.5278 0.0009  [0.525 0.526 0.527 0.527 0.528] | 0.527 \pm 0.001 | -3.4% \pm  0.1%
    MeLES embeddings:       
        Age group   |   0.639 \pm 0.006   | 0.6419  0.6372  0.6466 0.0034  [0.636 0.642 0.643 0.644 0.645] | 0.642 \pm 0.004 |  1.6% \pm  0.3%
        Gender      |   0.872 \pm 0.005   | 0.8821  0.8801  0.8840 0.0014  [0.880 0.882 0.882 0.882 0.884] | 0.882 \pm 0.002 |  0.3% \pm  0.2%
        Assessment  |   0.604 \pm 0.003   | 0.6041  0.5994  0.6088 0.0034  [0.598 0.604 0.605 0.606 0.607] | 0.604 \pm 0.004 |  2.2% \pm  0.3%
        Retail      |   0.544 \pm 0.001   | 0.5439  0.5421  0.5457 0.0013  [0.542 0.544 0.544 0.545 0.545] | 0.544 \pm 0.002 | -0.3% \pm  0.1%

Scores:                       
    Supervised learning:    
        Age group   |   0.631 \pm 0.010   | 0.6285  0.6172  0.6399 0.0082  [0.619 0.625 0.626 0.631 0.641] | 0.628 \pm 0.010 | -0.6% \pm  0.5%
        Gender      |   0.871 \pm 0.007   | 0.8741  0.8654  0.8828 0.0063  [0.865 0.872 0.873 0.879 0.881] | 0.874 \pm 0.008 | -0.6% \pm  0.3%
        Assessment  |   0.601 \pm 0.006   | 0.6010  0.5948  0.6072 0.0045  [0.596 0.600 0.600 0.601 0.608] | 0.601 \pm 0.005 |  1.7% \pm  0.3%
        Retail      |   0.543 \pm 0.002   | 0.5425  0.5403  0.5447 0.0016  [0.540 0.542 0.542 0.544 0.544] | 0.542 \pm 0.002 | -0.6% \pm  0.1%
    CPC fine-tuning:        
        Age group   |   0.621 \pm 0.007   | 0.6210  0.6166  0.6254 0.0032  [0.617 0.618 0.622 0.623 0.625] | 0.621 \pm 0.004 | -1.8% \pm  0.3%
        Gender      |   0.873 \pm 0.007   | 0.8777  0.8726  0.8829 0.0037  [0.874 0.875 0.878 0.879 0.883] | 0.878 \pm 0.004 | -0.2% \pm  0.2%
        Assessment  |   0.611 \pm 0.005   | 0.6115  0.6047  0.6184 0.0049  [0.603 0.611 0.613 0.615 0.616] | 0.612 \pm 0.006 |  3.5% \pm  0.4%
        Retail      |   0.546 \pm 0.003   | 0.5461  0.5429  0.5492 0.0023  [0.542 0.546 0.546 0.547 0.548] | 0.546 \pm 0.003 |  0.1% \pm  0.2%
    MeLES fine-tuning:      
        Age group   |   0.643 \pm 0.007   | 0.6383  0.6331  0.6435 0.0037  [0.632 0.637 0.641 0.641 0.641] | 0.638 \pm 0.005 |  1.0% \pm  0.3%
        Gender      |   0.888 \pm 0.002   | 0.8959  0.8910  0.9009 0.0036  [0.890 0.896 0.898 0.898 0.898] | 0.896 \pm 0.004 |  1.9% \pm  0.2%
        Assessment  |   0.614 \pm 0.003   | 0.6135  0.6095  0.6176 0.0029  [0.608 0.614 0.615 0.615 0.616] | 0.614 \pm 0.004 |  3.9% \pm  0.3%
        Retail      |   0.549 \pm 0.001   | 0.5490  0.5479  0.5500 0.0008  [0.548 0.549 0.549 0.550 0.550] | 0.549 \pm 0.001 |  0.7% \pm  0.1%

```
