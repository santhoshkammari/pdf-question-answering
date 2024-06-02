

##########################
# Install OLLAMA
#############################
# !pip install ollama
# ! sudo apt-get install -y pciutils
# ! curl https://ollama.ai/install.sh | sh


####################33

import os
import threading
import subprocess
import requests
import json

def ollama():
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])

ollama_thread = threading.Thread(target=ollama)
ollama_thread.start()




#################3

# !ollama pull llama3