# Example Sentiment Analysis with TensorFlow, LogisticRegression and CatBoost

This is an example project for sentiment analysis using TensorFlow. Sentiment analysis is the process of determining the sentiment or emotional tone of text, identifying whether it is positive, negative, neutral or irrelivant.

## Project Description

In this project, we utilize TensorFlow to build a machine learning model capable of classifying textual data based on sentiment. We will train the model on available datasets and evaluate its performance on new data.
We are going to compare results with LogisticRegression and CatBoost.

## Project goal

The goal of this project is to provide a structured example of sentiment analysis on Twitter data, especially for newcomers to the field. By sharing the code, data, and documentation, I am aiming to assist beginners in understanding the process of building and training a sentiment analysis model using TensorFlow.
Additionally, we strive to foster collaboration and learning within the NLP community.
Also, we want to campare different methods on one case in order to show and present them.

The models  performance results are presented in [ml_logs.txt](build/ml_logs.txt)

Please read the [MAQSAT](docs/MAQSAT.md) in order to learn author' vision of this project.

## Data

The data used for training and evaluating the sentiment analysis model is sourced from the following Kaggle dataset: - [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
Please refer to the dataset for more details about its contents and format. Make sure to comply with the dataset's license and terms of use when using the data.

## Project Structure

The project has the following structure:

```
- bin/
  - ready to go model
- build/
  - configs/
    - model YAML configs
  - ml_logs.txt (.tsv file with different models metrics)
  - labels.json (JSON file with label->number and number->label)
- data/
  - our csv files
- docs/
  - CODE_OF_CONDUCT.md
  - MAQSAT.md
  - TASKS.md
- jupyter_nbs/
  - (Jupyter notebooks go here)
- source/
  - (ready-to-use scripts go here)
```

## Project Setup

1. Clone the repository to your local machine:

```bash
git clone https://github.com/AigozhiyevB/twitter-nlp.git
```

2. Ensure that TensorFlow is installed. You can install it using pip:

```bash
pip install tensorflow
```

3. Navigate to the project directory:

```bash
cd twitter-nlp
```

4. Setup venv:

```bash
python -m venv *venv_name*
*venv_name*/bin/activate
```

5. Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the data for training the model. Place your data in the `data/` directory.

2. Run the Jupyter notebook to train the model based on the provided data. You will find the corresponding notebook in the `jupyter-nbs/` directory.

3. Execute the script from the `source/` directory to apply the trained model to new data and perform sentiment analysis.

## Contributing

If you would like to contribute to the project, you can do the following:

- Fork the repository.
- Make the necessary changes or add new features.
- Submit a pull request to have your changes merged into the main project branch.

More detailed instructions are presented in the [CONTRIBUTING](/CONTRIBUTING.md)

Where are plenty of tasks available in [TASKS](/docs/TASKS.md)

## About models

Short summaries of used models:

- [**TensorFlow**](https://www.tensorflow.org/): Open-source ML framework by _Google_, supports CPUs, GPUs, and specialized hardware. Provides high-level APIs and low-level operations.
- **Logistic regression**: Statistical binary classification model using the logistic function. Assumes linear relationship between features and class probabilities.
- [**CatBoost**](https://catboost.ai/): Gradient boosting framework by _Yandex_. Utilizes Ordered Boosting algorithm for effective handling of categorical features. Supports various tasks, GPU acceleration, and robustness to outliers.

## License

The project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

If you have any questions or suggestions, please contact me at [aigozhiyev.b@yandex.kz](mailto:aigozhiyev.b@yandex.kz?subject=[GitHub]%20Source%20twitter-nlp).

This project is developed in the Republic of Kazakhstan.
