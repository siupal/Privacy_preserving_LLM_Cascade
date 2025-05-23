@echo off
echo Installing required packages...

:: Install PyTorch (if not already installed)
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

echo Package installation complete!
