@echo off
echo Creating P3Defer environment...

:: Create new conda environment
call conda create -n p3defer python=3.10 -y

:: Activate environment
call conda activate p3defer

:: Install PyTorch (using conda to ensure CUDA support)
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

:: Install other dependencies
call pip install transformers
call pip install numpy
call pip install requests
call pip install python-Levenshtein
call pip install nltk
call pip install rouge-score
call pip install tqdm
call pip install matplotlib
call pip install pandas
call pip install jupyter

:: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo P3Defer environment setup complete!
echo Use 'conda activate p3defer' to activate the environment
