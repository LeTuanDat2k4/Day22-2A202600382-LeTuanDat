# RAGAS Evaluation Analysis: V1 vs V2

## Results Summary
* **Faithfulness**: V1 (0.7521) > V2 (0.5945)
* **Answer Relevancy**: V2 (0.8140) > V1 (0.7116)
* **Context Recall**: V1 (0.9600) == V2 (0.9600)
* **Context Precision**: V1 (0.8567) ≈ V2 (0.8533)

## Analysis
**Why did V1 score higher on Faithfulness?**
Prompt V1 explicitly instructs the LLM to use **"ONLY the provided context"** and keep the answer concise (2-4 sentences). This strict bounding limits the LLM's opportunity to hallucinate or bring in external training knowledge, resulting in higher faithfulness to the retrieved passages.

**Why did V2 score higher on Answer Relevancy?**
Prompt V2 asks the LLM to adopt an "expert AI tutor" persona and provide a "structured, well-organized answer." This leads to responses that are more comprehensive and directly address the user's underlying intent, scoring higher on relevancy. However, this same verbosity causes V2 to suffer in faithfulness, as it tends to elaborate beyond the strict boundaries of the retrieved text.

*Note: The overall Faithfulness slightly missed the 0.8 target (max 0.7521). This could be improved by fine-tuning the chunking strategy (e.g., semantic chunking instead of fixed-size) to ensure contexts contain more complete concepts, or by using a stronger generator LLM.*
