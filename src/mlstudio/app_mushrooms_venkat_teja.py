"""app_mushrooms_venkat_teja.py - my Phase 5 custom classification project.

Classify mushrooms as edible or poisonous from their physical characteristics,
and compare two classifiers: DecisionTreeClassifier vs RandomForestClassifier.

The UCI Mushroom dataset is fully separable, so with plenty of training data
both models score a perfect 1.00. To make the comparison meaningful, both
models are evaluated twice: once with a standard 70/30 split, and once
trained on only 40 mushrooms (0.5%) - scarce training data is where the
difference between a single tree and a forest becomes visible.

Author: Venkat Teja Nallamothu
Date: 2026-07

Process:
    - Load a CSV dataset.
    - Inspect and check data quality.
    - One-hot encode the categorical features.
    - Train two supervised classification models (decision tree, random forest).
    - Evaluate and compare both on a standard split and on scarce training data.
    - Create useful charts.

Data Source:
- data/raw/mushrooms.csv (UCI Machine Learning Repository, Mushroom dataset)

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_mushrooms_venkat_teja
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "mushrooms"

# STEP 1. Pick the target variable we want to predict.
# class is "p" (poisonous) or "e" (edible) in the raw data.

TARGET_COL: Final[str] = "class"
TARGET_LABELS: Final[dict[str, str]] = {"p": "poisonous", "e": "edible"}

# STEP 2. All other columns are categorical features (one letter codes).
# stalk_root uses "?" for missing values; it is kept as its own category
# because "unknown stalk root" may itself carry signal.

MISSING_CODE: Final[str] = "?"
MISSING_LABEL: Final[str] = "missing"

# STEP 3. Define the test size and random state for reproducibility.
# SMALL_TRAIN_SIZE trains on only ~40 mushrooms (0.5%) for the second
# comparison scenario: the dataset is fully separable, so both models are
# perfect on a standard split, and scarce training data is what reveals
# the generalization difference between one tree and a forest.

TEST_SIZE: Final[float] = 0.30
SMALL_TRAIN_SIZE: Final[float] = 0.005
RANDOM_STATE: Final[int] = 42

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the mushrooms dataset from the data/raw folder."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    LOG.info(f"Class balance:\n{df[TARGET_COL].value_counts()}")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column (as NaN)")
    LOG.debug(f"\n{df.isna().sum()}")

    missing_code_count: int = int((df == MISSING_CODE).sum().sum())
    LOG.info(f"Cells coded '{MISSING_CODE}' (missing): {missing_code_count}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")


# === Section 5. Create a Clean, Encoded View ===


def make_encoded_view(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the modeling view: readable labels and one-hot encoded features.

    Trees can split on categories, but scikit-learn requires numeric input,
    so every categorical feature is one-hot encoded (one 0/1 column per value).
    """
    LOG.info("Creating encoded modeling view")

    df_clean: pd.DataFrame = df.replace(MISSING_CODE, MISSING_LABEL).copy()

    y: pd.Series = df_clean[TARGET_COL].map(TARGET_LABELS)  # type: ignore[assignment, arg-type]
    x_raw: pd.DataFrame = df_clean.drop(columns=[TARGET_COL])

    x_encoded: pd.DataFrame = pd.get_dummies(x_raw, prefix_sep="=")

    LOG.info(
        f"Encoded view: {x_encoded.shape[0]} rows, "
        f"{x_raw.shape[1]} features -> {x_encoded.shape[1]} one-hot columns"
    )
    return x_encoded, y


# === Section 6. Train and Compare Supervised Models ===


def evaluate_model(
    name: str,
    model: DecisionTreeClassifier | RandomForestClassifier,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> float:
    """Fit one model and log its evaluation metrics; return test accuracy."""
    LOG.info(f"--- {name} ---")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy: float = float(accuracy_score(y_test, y_pred))
    LOG.info(f"{name} accuracy: {accuracy:.4f}")

    LOG.info(f"{name} confusion matrix (rows=true, cols=predicted)")
    LOG.info(f"\n{confusion_matrix(y_test, y_pred)}")

    LOG.info(f"{name} classification report")
    LOG.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")  # type: ignore[arg-type]

    return accuracy


def compare_models(
    x: pd.DataFrame,
    y: pd.Series,
    scenario: str,
    train_size: float | None = None,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train a decision tree and a random forest on the same split and compare.

    By default the split holds back TEST_SIZE for testing. Passing train_size
    instead trains on that small fraction and tests on everything else.
    """
    LOG.info("========================")
    LOG.info(f"COMPARE MODELS: {scenario}")
    LOG.info("========================")

    # Declare split types once; train_test_split's stub returns an untyped list.
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    if train_size is None:
        x_train, x_test, y_train, y_test = train_test_split(  # type: ignore[assignment]
            x,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(  # type: ignore[assignment]
            x,
            y,
            train_size=train_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    LOG.info(f"Training rows: {len(x_train)}, test rows: {len(x_test)}")

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    results: dict[str, float] = {
        "Decision Tree": evaluate_model(
            "Decision Tree", tree, x_train, x_test, y_train, y_test
        ),
        "Random Forest": evaluate_model(
            "Random Forest", forest, x_train, x_test, y_train, y_test
        ),
    }

    return forest, results


# === Section 7. Predict One New Case ===


def predict_example(model: RandomForestClassifier, x_encoded: pd.DataFrame) -> None:
    """Use the trained forest to predict one held-out style example."""
    LOG.info("Predicting one new case")

    # Reuse the first row's encoding as a stand-in "new" mushroom.
    new_case: pd.DataFrame = x_encoded.iloc[[0]]

    predicted_label: str = model.predict(new_case)[0]
    LOG.info(f"Predicted result for the sample mushroom: {predicted_label}")


# === Section 8. Create Visualizations ===


def make_plots(
    forest: RandomForestClassifier,
    x_encoded: pd.DataFrame,
    results_full: dict[str, float],
    results_scarce: dict[str, float],
) -> None:
    """Create charts for the model comparison."""
    LOG.info("Creating chart: top 10 random forest feature importances")

    importance_df = (
        pd.DataFrame(
            {
                "feature": x_encoded.columns,
                "importance": forest.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_plt: Axes = sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        ax=ax,
    )

    bar_plt.set_title(
        "Top 10 Feature Importances - Random Forest (CLOSE chart to continue)"
    )
    bar_plt.set_xlabel("Importance")
    bar_plt.set_ylabel("Feature (one-hot column)")

    LOG.info("Creating chart: accuracy comparison")

    comparison_df = pd.DataFrame(
        [
            {"scenario": "70/30 split", "model": model, "accuracy": acc}
            for model, acc in results_full.items()
        ]
        + [
            {"scenario": "40 training rows", "model": model, "accuracy": acc}
            for model, acc in results_scarce.items()
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    compare_plt: Axes = sns.barplot(
        data=comparison_df,
        x="scenario",
        y="accuracy",
        hue="model",
        ax=ax,
    )

    compare_plt.set_title("Decision Tree vs Random Forest (CLOSE chart to continue)")
    compare_plt.set_xlabel("Scenario")
    compare_plt.set_ylabel("Test Accuracy")
    compare_plt.set_ylim(0.85, 1.0)


# === Section 9. Summary and Next Steps ===


def summarize(
    df: pd.DataFrame,
    results_full: dict[str, float],
    results_scarce: dict[str, float],
) -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Rows: {df.shape[0]}")
    LOG.info(f"Target: {TARGET_COL} (edible vs poisonous)")
    for model, acc in results_full.items():
        LOG.info(f"70/30 split      | {model}: {acc:.4f}")
    for model, acc in results_scarce.items():
        LOG.info(f"40 training rows | {model}: {acc:.4f}")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the supervised ML workflow."""
    log_header(LOG, "ML")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Encode features...........")
    x_encoded, y = make_encoded_view(df)

    LOG.info("Compare models (70/30 split)......")
    forest, results_full = compare_models(x_encoded, y, "standard 70/30 split")

    LOG.info("Compare models (scarce training)..")
    _, results_scarce = compare_models(
        x_encoded, y, "scarce training data", train_size=SMALL_TRAIN_SIZE
    )

    LOG.info("Predict one case..........")
    predict_example(forest, x_encoded)

    LOG.info("Create charts.............")
    make_plots(forest, x_encoded, results_full, results_scarce)

    LOG.info("Summarize workflow........")
    summarize(df, results_full, results_scarce)

    LOG.info(
        "----- in a script, call plt.show() once at the end to display all charts -----"
    )
    LOG.info(
        "----- in a script, CLOSE the chart windows with the close button to CONTINUE -----"
    )

    plt.show()

    LOG.info("Workflow complete")
    LOG.info("IMPORTANT: This script creates chart windows.")
    LOG.info("Close chart windows and terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

# WHY: Only call main() when running this file directly as a script.
# This is standard Python boilerplate.

if __name__ == "__main__":
    main()
