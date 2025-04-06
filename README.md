# UNLP Kaggle Competition Research Report

This repository contains my experimental work for the [UNLP Challenge](https://www.kaggle.com/competitions/unlp-2025-shared-task-classification-techniques/overview), a text classification competition on Kaggle focused on identifying manipulation techniques in textual content.

## Approach

I explored multiple classification approaches, focusing on both traditional NLP methods and modern LLM-based techniques. The primary goal was to determine the most effective approach given resource constraints and the nature of the classification problem, particularly considering class imbalance issues.

## Techniques and Results

| Approach | Description | F1 Score |
|----------|-------------|----------|
| XML-RoBERTa (large) | Fine-tuned transformer model with classification head, best handling of imbalanced classes | 0.392 |
| XML-RoBERTa (base) | Smaller version of XML-RoBERTa with classification head | 0.368 |
| TF-IDF with Linear Regression | Traditional NLP approach enhanced with synthetic data for minority classes generated using Mistral Large | 0.359 |
| TF-IDF with SVM | Classical machine learning approach (synthetic data actually reduced performance) | 0.310 |
| RAG with Mistral Nemo | Retrieval-augmented generation searching for similar labeled messages from training data | 0.309 |
| Mistral Nemo Fine-tuning | Fine-tuned on training dataset using question-answer format | 0.280 |
| Gemma 1B Fine-tuning | Fine-tuned using question-answer format | 0.280 |

## Key Findings

1. **Best Performer**: XML-RoBERTa (large) outperformed other approaches, likely due to its pre-training on cross-lingual data and better handling of imbalanced classes.

2. **Traditional NLP**: TF-IDF based methods performed surprisingly well, particularly when augmented with synthetic data for minority classes. However, synthetic data generation proved to be complex and requires careful implementation, in my case it was not very well quality data.

3. **Small LLMs**: Models like Gemma 1B underperformed, primarily due to not being optimized for classification tasks. These models frequently produced hallucinations and irrelevant outputs.

4. **RAG Approach**: While innovative, the retrieval-augmented generation approach with Mistral Nemo didn't achieve competitive results compared to fine-tuned transformer models.

## Resources and Limitations

The experiments were conducted using:
- Google Colab free tier
- Kaggle free tier notebooks

These computational constraints influenced model selection and training strategies, potentially limiting the performance of more resource-intensive approaches.

## Conclusion

XML-RoBERTa large proved to be the most effective model for this classification task, demonstrating superior performance in handling imbalanced classes. Traditional TF-IDF approaches remain competitive when properly implemented, especially when enhanced with synthetic data. Small language models showed significant limitations for classification tasks in this context.

Future work could explore ensemble methods combining the strengths of different approaches or more sophisticated data augmentation techniques to address class imbalance issues.
