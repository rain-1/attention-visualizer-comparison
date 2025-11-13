# Attention Visualizer Comparison

This project provides a small Flask application for comparing attention maps stored in two PyTorch `.pt` files. The UI renders side-by-side heatmaps for each model so you can inspect differences across layers and heads.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place your serialized attention files in `app/data/` (or set the `MODEL_A_PATH` / `MODEL_B_PATH` environment variables). The tensors must be stored with the following structure:

   ```python
   torch.save({
       "tokens": tokens,                 # list of string tokens
       "attention": attention_tensor,    # shape [layers, heads, tokens, tokens]
       "prompt": prompt_text,
       "num_layers": attention_tensor.shape[0],
       "num_heads": attention_tensor.shape[1],
   }, "model_a.pt")
   ```

3. Launch the development server:

   ```bash
   export MODEL_A_PATH=app/data/model_a.pt
   export MODEL_B_PATH=app/data/model_b.pt
   python -m app.server
   ```

4. Open your browser to `http://localhost:8000` to explore the attention comparison view. Use the head slider to select which attention head to inspect and the layer slider to page through layers eight at a time.

## Development notes

- Heatmaps are rendered with Plotly to provide responsive SVG/Canvas output that works well for long token lists.
- Data from the models is cached in memory for quick navigation between layers and heads.

## Sample data

You can generate synthetic attention tensors for experimentation by running:

```bash
python scripts/generate_sample_data.py
```

This will create two toy models in `app/data/` that can be visualized immediately.
