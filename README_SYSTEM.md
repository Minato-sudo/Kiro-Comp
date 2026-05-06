# AI Resume & Job Matching System (Research-Grade)

This system implements a state-of-the-art dual-encoder architecture for semantically matching resumes to job descriptions, incorporating 6 novel research-level contributions.

## Key Novelties
1.  **N1: Asymmetric Cross-Attention**: JD requirements query resume sections to find deep supporting evidence.
2.  **N2: Dynamic Skill Graph**: Automatically extends the ESCO taxonomy based on skill co-occurrences in JDs.
3.  **N3: LLM as Weak Labeller**: Distills knowledge from large models to train specialized lightweight encoders.
4.  **N4: Score Calibration**: Uses Platt scaling to convert raw similarity into interpretable match probabilities.
5.  **N5: Counterfactual Rewriting**: Calculates the exact "delta-score" impact of adding specific bullet points.
6.  **N6: Bias Audit**: Automated checks for gender and age-related scoring disparities.

## Project Structure
- `app.py`: Main Streamlit dashboard.
- `src/model.py`: PyTorch implementation of the Dual Encoder and Cross-Attention.
- `src/pipeline.py`: PDF/DOCX parsing and coordination.
- `src/ontology.py`: Dynamic skill graph logic.
- `src/explainability.py`: LLM integration for rewrites and suggestions.
- `src/bias_audit.py`: Fairness and bias auditing tools.

## How to Run
1.  **Activate Virtual Environment**:
    ```bash
    source venv/bin/activate
    ```
2.  **Run Streamlit**:
    ```bash
    streamlit run app.py
    ```

## Dependencies
- `torch`, `transformers`, `sentence-transformers`
- `PyMuPDF` (fitz), `python-docx`
- `streamlit`, `plotly`, `networkx`
- `groq` (Optional for LLM features)

*Note: For full LLM features (Rewriting/Suggestions), set your `GROQ_API_KEY` in environment variables.*
