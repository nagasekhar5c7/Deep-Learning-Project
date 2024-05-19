Below is a `README.md` file for your Deep-Learning-Project:


# Deep-Learning-Project

This project focuses on processing a dataset of heart patient information, generating correlation graphs, and fine-tuning a Large Language Model (LLM) using the Llama 2 architecture.

## Steps to Follow:

### 1. Download the Dataset
Access the provided Google Drive link to download the dataset files to your local machine.

### 2. Prepare the Environment
Ensure you have the following installed on your system:
 - Python
 - accelerate==0.21.0
 - peft==0.4.0
 - bitsandbytes==0.40.2
 - transformers==4.31.0
 - trl==0.4.7

You can install these dependencies using pip:
```bash
pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
```

### 3. Generate Pickle Files
1. Locate the `main.py` script in the project directory.
2. Run the `main.py` script to generate the pickle files. This script processes the dataset and creates files containing heart patient information and a correlation graph.
```bash
python main.py
```

### 4. Run the Fine-tune Llama 2 Notebook
1. Ensure you have Jupyter Notebook or JupyterLab installed. If not, install it using:
```bash
pip install notebook
```
2. Open Jupyter Notebook by running the following command in your terminal:
```bash
jupyter notebook
```
3. Navigate to the `Fine_tine_Llama_2.ipynb` file and open it.
4. Execute the cells in the notebook sequentially to fine-tune the LLM using the generated training and testing files.

