# SQL Assistant

A fine-tuned language model for generating SQL queries from natural language questions. This project fine-tunes the Qwen2.5-1.5B-Instruct model using Parameter-Efficient Fine-Tuning (LoRA) on SQL generation tasks.

## Project Overview

SQL Assistant is designed to help users generate SQL queries from natural language questions given a database schema context. The model is fine-tuned on the `b-mc2/sql-create-context` dataset, which contains SQL CREATE TABLE statements paired with questions and their corresponding SQL answers.

### Key Features

- **Parameter-Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) to fine-tune a large language model efficiently
- **4-bit Quantization**: Utilizes BitsAndBytesConfig for memory-efficient training and inference
- **Chat Template Format**: Follows Qwen's chat template format for structured conversations
- **SQL Generation**: Converts natural language questions to SQL queries given table schemas

## Repository Structure

```
SQL-Assistant/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── sql_assistant.ipynb       # Training and evaluation notebook
└── results/                  # Training output directory 
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training and inference)
- Hugging Face account and access token (for model downloads and uploads)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/MANU-de/SQL-Assistant.git
   cd SQL-Assistant
   ```

2. **Create Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   If you prefer to install manually, you can install the required packages:

   ```bash
   pip install -q -U torch transformers datasets peft bitsandbytes accelerate "numpy<2.0"
   pip install -q -U trl==0.9.6
   pip install -q huggingface_hub
   ```

4. **Configure Hugging Face Authentication**:

   - Get your Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - In the notebook (Cell 0), replace `my_token = ""` with your token:
     ```python
     my_token = "your_huggingface_token_here"
     ```

## How to Reproduce

### Training the Model

1. **Open the Jupyter Notebook**:

   ```bash
   jupyter notebook sql_assistant.ipynb
   ```

2. **Run Training Cells** (Cell 0 and Cell 1):

   - **Cell 0**: Installs dependencies and authenticates with Hugging Face
   - **Cell 1**: 
     - Loads the base model (`Qwen/Qwen2.5-1.5B-Instruct`)
     - Loads and prepares the dataset (`b-mc2/sql-create-context`, 1000 examples)
     - Configures 4-bit quantization and LoRA parameters
     - Trains the model and saves the fine-tuned adapter

   The training process will:
   - Download the base model and dataset (first run only)
   - Fine-tune the model using LoRA
   - Save the adapter to `./Qwen2.5-1.5B-SQL-Assistant/`

### Evaluation and Inference

1. **Run Evaluation Cell** (Cell 2):

   - Loads the base model and fine-tuned adapter
   - Tests the model with a sample query
   - Generates SQL from a natural language question

   Example test case:
   ```python
   context = "CREATE TABLE employees (employee_id INT PRIMARY KEY, name VARCHAR(255) NOT NULL, role VARCHAR(255), manager_id INT, FOREIGN KEY (manager_id) REFERENCES employees(employee_id))"
   question = "Which employees report to the manager 'Julia König'?"
   ```

### Expected Results

After training, you should see:
- Training loss logged every 10 steps
- A saved model directory: `Qwen2.5-1.5B-SQL-Assistant/`
- Generated SQL queries that correspond to the input questions and contexts

## Configuration Details

### Model Configuration

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct` (1.5B parameters)
- **Fine-tuned Model Name**: `Qwen2.5-1.5B-SQL-Assistant`
- **Dataset**: `b-mc2/sql-create-context` (1000 samples used for demo)

### Training Configuration

- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Parameters**:
  - `r=16` (rank)
  - `lora_alpha=16`
  - `lora_dropout=0.05`
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Training Arguments**:
  - Batch size: 4 per device
  - Gradient accumulation: 2 steps
  - Learning rate: 2e-4
  - Epochs: 1
  - Mixed precision: FP16
  - Optimizer: paged_adamw_32bit

### Quantization

- **4-bit Quantization**: Enabled for memory efficiency
- **Quantization Type**: NF4
- **Compute Dtype**: float16

## Model Card

### Model Information

- **Model Name**: Qwen2.5-1.5B-SQL-Assistant
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning Method**: LoRA (Parameter-Efficient Fine-Tuning)
- **Task**: Text-to-SQL generation

### Intended Use

The model is designed to:
- Generate SQL queries from natural language questions
- Work with provided database schema contexts (CREATE TABLE statements)
- Assist developers and data analysts in writing SQL queries

### Training Data

- **Dataset**: b-mc2/sql-create-context
- **Samples Used**: 1000 (demo subset)
- **Data Format**: SQL CREATE TABLE contexts paired with questions and SQL answers
- **Data Split**: Training set only (no validation/test split in current implementation)

### Limitations

- **Limited Training Data**: Currently uses only 1000 samples from the full dataset
- **Single Epoch**: Model is trained for only 1 epoch (may benefit from additional training)
- **No Evaluation Metrics**: Current implementation focuses on demonstration rather than comprehensive evaluation
- **Dialect Coverage**: Performance may vary across different SQL dialects
- **Context Length**: Limited to 512 tokens maximum sequence length
- **GPU Required**: Training and inference require CUDA-capable GPU for reasonable performance

### Performance Considerations

- The model uses 4-bit quantization, which reduces memory usage but may slightly impact accuracy
- Training on a full dataset with multiple epochs and proper validation would improve performance
- Evaluation on standard benchmarks (e.g., Spider, WikiSQL) would provide better performance metrics

### Ethical Considerations

- The model should be used responsibly and its outputs should be validated before execution on production databases
- SQL queries should be tested in a safe environment before applying to real databases
- Be mindful of potential security implications (SQL injection risks)

## Code Quality

### Reproducibility

- Fixed random seed (seed=42) for dataset shuffling
- All dependencies are specified with versions in `requirements.txt`
- Training configurations are clearly documented in the notebook
- Model and dataset identifiers are explicitly defined

### Code Organization

- **Clean Structure**: Code is organized into logical cells (setup, training, evaluation)
- **Comments**: Key steps are commented in English (some German comments in original code)
- **Modularity**: Functions are defined for reusable components (e.g., `format_prompt`)

### Environment-Specific Configurations

- **Hugging Face Token**: Required for model downloads/uploads (set in Cell 0)
- **CUDA Device**: Code assumes CUDA availability (`device_map="auto"`, `.to("cuda")`)
- **Memory Constraints**: 4-bit quantization used to reduce memory requirements
- **Dataset Size**: Currently limited to 1000 samples (can be adjusted in Cell 1)

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce `per_device_train_batch_size` or `max_seq_length`
   - Ensure 4-bit quantization is enabled
   - Restart kernel between training and evaluation

2. **Hugging Face Authentication Error**:
   - Verify your token is correctly set in Cell 0
   - Check token permissions (read access for models, write for uploads)

3. **CUDA Not Available**:
   - Verify GPU is detected: `torch.cuda.is_available()`
   - Install appropriate CUDA toolkit if needed
   - Consider using CPU (much slower, not recommended for training)

4. **Model Download Issues**:
   - Check internet connection
   - Verify Hugging Face model accessibility
   - Ensure sufficient disk space for model downloads (~3GB for base model)

## Future Improvements

- [ ] Add proper train/validation/test splits
- [ ] Implement comprehensive evaluation metrics
- [ ] Train on full dataset with multiple epochs
- [ ] Add support for different SQL dialects
- [ ] Create evaluation script for standard benchmarks
- [ ] Add support for multiple table contexts
- [ ] Implement model upload to Hugging Face Hub

## License

This project is open source. Please refer to the license of the base model (Qwen2.5-1.5B-Instruct) and dataset (b-mc2/sql-create-context) for usage terms.

## Acknowledgments

- **Base Model**: Qwen Team for Qwen2.5-1.5B-Instruct
- **Dataset**: b-mc2/sql-create-context dataset contributors
- **Libraries**: Hugging Face Transformers, PEFT, TRL, and BitsAndBytes teams

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/MANU-de/SQL-Assistant).

---

*This README follows best practices for ML project documentation and aims to ensure reproducibility and clarity.*

