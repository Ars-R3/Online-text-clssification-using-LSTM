Online Text Classification using LSTM
====================================

Multi-label text classification of online comments using an LSTM model built with Keras/TensorFlow. The repository contains a single Jupyter notebook that provides an interactive, menu-driven workflow for training, evaluation, inference, and model saving.

- Original repository: [Ars-R3/Online-text-clssification-using-LSTM](https://github.com/Ars-R3/Online-text-clssification-using-LSTM)


Project structure
-----------------

- `Mid_LSTM_project.ipynb` — main notebook with an interactive menu.
- `index.html` — static presentation page (open in your browser).
- `styles.css` — styles for the presentation page.


Static presentation frontend
----------------------------

You can open the static presentation by simply opening `index.html` in your web browser.

Optionally, serve it locally for consistent asset loading:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/Online-text-clssification-using-LSTM/` (adjust the path if serving from a different directory).


Requirements
------------

- Python 3.9+ recommended
- Packages:
  - tensorflow
  - pandas
  - numpy
  - scikit-learn
  - jupyter (to run the notebook)

Install (venv + pip)
--------------------

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow pandas numpy scikit-learn jupyter
```

Alternatively (conda):

```bash
conda create -n lstm-text python=3.10 -y
conda activate lstm-text
pip install tensorflow pandas numpy scikit-learn jupyter
```


Data format
-----------

Training CSV must contain the following columns:

- `comment_text` — the raw text input
- Label columns (multi-label, 0/1): `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

An optional test CSV should at least contain `comment_text` when using the prediction workflow.


How to run
----------

1) Launch Jupyter and open the notebook:

```bash
jupyter notebook
```

Open `Mid_LSTM_project.ipynb` and run the first cell. You will see a printed menu:

```
####################################################################
                 classifying online toxic comments
####################################################################
Choose one of the following options to continue:
1- Classify dataset of comments using LSTM model
2- Split data and find the accuracy
3- classify  dataset of comments  using keras
4- load a pretrain keras model
5- Classify a single comment using LSTM model
--------------------------------------------------------------------
```

Menu options (what they actually do)
------------------------------------

- 1) Train on train CSV and generate predictions for a separate test CSV
  - Prompts for train and test CSV paths.
  - Trains an LSTM with embedding → LSTM → GlobalMaxPooling1D → Dense layers.
  - Writes a predictions CSV (you choose the output file name) and prints a summary count for each category using a 0.5 threshold.

- 2) Train on train CSV and predict a single ad-hoc comment
  - Prompts for train CSV path and a comment string.
  - Trains the model on the entire training data, then predicts for the single provided comment and prints the raw probabilities.

- 3) Train/test split and accuracy
  - Prompts for train CSV path.
  - Splits the preprocessed data into train/test (80/20) and reports accuracy using a 0.5 threshold on predictions.

- 4) Build and save an untrained Keras model
  - Builds an LSTM model with `input_length=200` and `num_classes=6` and saves it as a `.keras` file (you choose the filename).

- 5) Load a saved Keras model and predict a single comment
  - Prompts for a `.keras` model path and a comment string; prints prediction probabilities.


Notes and tips
--------------

- Tokenization uses a vocabulary size of `20000` and sequence length `200`.
- Empty/zero-length tokenized comments are filtered out automatically.
- Default training uses `epochs=10`; batch size is `256` (option 1/3) or `8192` (option 2). If you face memory issues, reduce `batch_size`.
- For faster training, consider using a GPU-enabled TensorFlow build.


Example command-line workflow (optional)
---------------------------------------

If you prefer running programmatically, open the notebook and run cells in order; the workflow relies on `input()` prompts within the notebook interface.


License
-------

No license file was found in the original repository. Review the upstream repo and add a license if needed.

Contributors
------------

- Arslan Ilyas


