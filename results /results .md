# Evaluation Results

This document presents the evaluation results and performance analysis of the fine-tuned **Qwen2.5-1.5B-SQL-Assistant** model.

## Model Information

- **Model Name**: Qwen2.5-1.5B-SQL-Assistant
- **Hugging Face Model Hub**: [manuelaschrittwieser/Qwen2.5-1.5B-SQL-Assistant](https://huggingface.co/manuelaschrittwieser/Qwen2.5-1.5B-SQL-Assistant)
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with 4-bit quantization (QLoRA)
- **Training Dataset**: b-mc2/sql-create-context

## Training Results

### Training Visualization

<img width="588" height="395" alt="Screenshot 2025-12-04 12 49 07 PM" src="https://github.com/user-attachments/assets/b044ec36-4764-4e37-a291-01fad0b98127" />

*Training metrics and loss curves showing the model's performance during fine-tuning.*

### Training Configuration Summary

- **Learning Rate**: 2e-4
- **Batch Size**: 4 per device (with gradient accumulation)
- **Epochs**: 1
- **Optimizer**: paged_adamw_32bit
- **LoRA Configuration**:
  - Rank (r): 16
  - LoRA Alpha: 16
  - LoRA Dropout: 0.05
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Hardware**: NVIDIA T4 GPU
- **Quantization**: 4-bit NF4 quantization for memory efficiency

### Training Monitoring

- **Weights & Biases Dashboard**:

  <img width="1813" height="770" alt="Screenshot 2025-12-04 1 33 31 PM" src="https://github.com/user-attachments/assets/3e9db663-2734-437a-8929-25441c89d43a" />

The training was monitored in real-time using Weights & Biases, providing insights into:
- Training loss progression
- Learning rate scheduling
- System resource utilization
- Model performance metrics

## Evaluation Results

### Baseline vs. Fine-Tuned Comparison

The following table compares the performance of the base model (Qwen 2.5-1.5B-Instruct) with the fine-tuned SQL Assistant model:

| Feature              | Base Model (Qwen 2.5-1.5B-Instruct)                      | Fine-Tuned Model (SQL-Assistant)                       |
| -------------------- | -------------------------------------------------------- | ------------------------------------------------------ |
| **Response Format**  | Often chatty; explains the code before/after.            | Concise; outputs strictly the SQL query.               |
| **Schema Adherence** | Sometimes hallucinates column names not in the schema.   | Strongly adheres to the provided CREATE TABLE context. |
| **Syntax Accuracy**  | Good, but prone to minor syntax errors in complex joins. | Improved syntax specific to standard SQL queries.      |
| **Task Focus**       | General-purpose instruction following                    | Specialized for SQL generation                         |

### Key Improvements

1. **Format Consistency**: The fine-tuned model produces clean, executable SQL queries without verbose explanations
2. **Schema Awareness**: Better understanding of provided table schemas, reducing hallucination of non-existent columns
3. **SQL Syntax**: More accurate SQL syntax, especially in complex query structures

## Sample Test Cases

### Example 1: Basic Query with WHERE Clause

**Input Context:**
```sql
CREATE TABLE employees (name VARCHAR, dept VARCHAR, salary INT)
```

**Question:**
> "Who works in Sales and earns more than 50k?"

**Model Output:**
```sql
SELECT name FROM employees WHERE dept = 'Sales' AND salary > 50000
```

### Example 2: Complex Query with JOIN

**Input Context:**
```sql
CREATE TABLE employees (employee_id INT PRIMARY KEY, name VARCHAR(255) NOT NULL, role VARCHAR(255), manager_id INT, FOREIGN KEY (manager_id) REFERENCES employees(employee_id))
```

**Question:**
> "Which employees report to the manager 'Julia König'?"

**Model Output:**
The model generates a SQL query that correctly:
- Joins the employees table with itself (self-join)
- Filters by manager name
- Returns the appropriate employee information

### Example 3: Students Query

**Input Context:**
```sql
CREATE TABLE students (id INT, name VARCHAR, grade INT, subject VARCHAR)
```

**Question:**
> "List the names of students in grade 10 who study Math."

**Model Output:**
```sql
SELECT name FROM students WHERE grade = 10 AND subject = 'Math'
```

## Quantitative Performance

### Training Metrics

- **Training Loss**: Monitored and logged every 10 steps
- **Convergence**: Model shows stable loss reduction throughout training
- **Memory Efficiency**: 4-bit quantization enabled training on consumer-grade GPUs

### Model Size

- **Base Model**: 1.5B parameters
- **LoRA Adapters**: ~16M parameters (significantly smaller than full fine-tuning)
- **Memory Savings**: Approximately 4x reduction in memory usage through quantization

## Limitations and Known Issues

### Current Limitations

1. **Scope**: The model is specialized for SQL generation. Performance on general creative writing or open-ended chat tasks may be reduced compared to the base model.

2. **Context Dependency**: 
   - The model relies heavily on the provided schema context
   - If column names are ambiguous or missing from the context, the model may fail or hallucinate

3. **Query Complexity**:
   - Effective for standard queries (SELECT, JOIN, WHERE, GROUP BY)
   - May struggle with extremely complex nested sub-queries
   - Limited support for database-specific proprietary functions (e.g., Oracle/Postgres extensions)

4. **Training Data**:
   - Currently trained on 1000 samples (subset of full dataset)
   - Single epoch training may benefit from additional epochs
   - No formal validation/test split in current evaluation

5. **SQL Dialect Coverage**:
   - Performance may vary across different SQL dialects
   - Primarily optimized for standard SQL syntax

## Future Evaluation Recommendations

To further improve evaluation and understanding of model performance, consider:

1. **Standard Benchmarks**:
   - Evaluate on Spider dataset for text-to-SQL tasks
   - Test on WikiSQL for simple SQL generation
   - Benchmark on more complex SQL datasets

2. **Metrics**:
   - Execution accuracy (does the generated SQL execute correctly?)
   - Exact match accuracy (does it match the ground truth exactly?)
   - Schema linking accuracy
   - Query complexity analysis

3. **Ablation Studies**:
   - Compare different LoRA configurations
   - Evaluate impact of training data size
   - Test different prompt formatting strategies

4. **Cross-Dialect Evaluation**:
   - Test performance across PostgreSQL, MySQL, SQLite
   - Evaluate dialect-specific features

## Resources

- **Model Card**: [Hugging Face Model Card](https://huggingface.co/manuelaschrittwieser/Qwen2.5-1.5B-SQL-Assistant)
- **Training Dashboard**: *see screenshot (Weights & Biases Dashboard:)*
- **Dataset**: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
- **Base Model**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

## Conclusion

The fine-tuned Qwen2.5-1.5B-SQL-Assistant model demonstrates significant improvements over the base model for SQL generation tasks. The model shows:

- ✅ Better adherence to provided database schemas
- ✅ More concise and executable SQL output
- ✅ Improved syntax accuracy for standard SQL queries
- ✅ Efficient training using parameter-efficient fine-tuning

While the model has limitations in handling very complex queries and database-specific features, it provides a solid foundation for text-to-SQL applications and can be further improved with additional training data and epochs.

---

*For usage instructions and code examples, please refer to the [model card on Hugging Face](https://huggingface.co/manuelaschrittwieser/Qwen2.5-1.5B-SQL-Assistant) or the main [README.md](README.md).*
