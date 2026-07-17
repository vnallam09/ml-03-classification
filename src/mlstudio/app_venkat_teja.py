"""app_venkat_teja.py - my copy of the example (Phase 4 modification).

A supervised classification case, copied from app_case.py.

Phase 4 change:
    Converted the regression example into a classification case - the theme of
    this module. A new derived target, `passed` ("pass" if score >= 75, else
    "fail"), replaces the numeric score, and a DecisionTreeClassifier replaces
    LinearRegression. The model is evaluated with accuracy, a confusion matrix,
    and a classification report (precision / recall / F1) instead of MAE and
    R-squared, and the coefficient chart is replaced with a confusion matrix
    heatmap.

Author: Venkat Teja Nallamothu
Date: 2026-07

Process:
    - Load a CSV dataset.
    - Derive a categorical target (pass/fail).
    - Train a supervised classification model.
    - Evaluate model performance.
    - Predict one new case.
    - Create useful charts.

Data Source:
- data/raw/hours_scores_case.csv

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_venkat_teja
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "hours_scores_case"

# STEP 1. Pick the target variable we want to predict.
# Phase 4 change: the target is a derived category, not the numeric score.
# A student with score >= PASS_THRESHOLD is labeled "pass", otherwise "fail".

SCORE_COL: Final[str] = "score"
TARGET_COL: Final[str] = "passed"
PASS_THRESHOLD: Final[int] = 75

# STEP 2. Define the column names (features) that may help predict the target variable.

FEATURE_COLS: Final[list[str]] = [
    "hours_studied",
    "practice_quizzes",
    "attendance_pct",
    "sleep_hours",
    "prior_score",
]

# STEP 3. Define the test size and random state for reproducibility.

TEST_SIZE: Final[float] = 0.30
RANDOM_STATE: Final[int] = 42

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the case dataset from the data/raw folder."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 2b. Derive the Categorical Target (Phase 4 modification) ===


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the pass/fail target from the numeric score.

    passed = "pass" if score >= PASS_THRESHOLD else "fail"
    Turning the continuous score into a category converts this
    regression example into a classification problem.
    """
    LOG.info(f"Deriving target: {TARGET_COL} = {SCORE_COL} >= {PASS_THRESHOLD}")

    df[TARGET_COL] = df[SCORE_COL].apply(
        lambda s: "pass" if s >= PASS_THRESHOLD else "fail"
    )

    LOG.info(f"Class balance:\n{df[TARGET_COL].value_counts()}")
    LOG.debug(f"\n{df[[SCORE_COL, TARGET_COL]]}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info("DataFrame info")
    df.info()

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    LOG.debug(f"\n{df.isna().sum()}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")


# === Section 5. Create a Clean View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view for modeling."""
    LOG.info("Creating clean modeling view")

    selected_cols: list[str] = FEATURE_COLS + [TARGET_COL]

    # Select only the columns we need.
    df_selected: pd.DataFrame = df[selected_cols]  # type: ignore[assignment]

    # Drop rows with any missing values.
    df_no_missing: pd.DataFrame = df_selected.dropna()

    # Assign a copy of the no-missing DataFrame to df_clean to avoid SettingWithCopyWarning.
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean


# === Section 6. Train Supervised Model ===


def train_model(df_clean: pd.DataFrame) -> DecisionTreeClassifier:
    """Train a supervised classification model."""
    LOG.info("Training DecisionTreeClassifier model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    # stratify=y keeps the pass/fail ratio similar in both splits,
    # which matters on a dataset this small.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy: float = float(accuracy_score(y_test, y_pred))
    LOG.info(f"Accuracy: {accuracy:.2f}")

    LOG.info("Confusion matrix (rows=true, cols=predicted)")
    LOG.info(f"\n{confusion_matrix(y_test, y_pred)}")

    LOG.info("Classification report")
    LOG.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")  # type: ignore[arg-type]

    return model


# === Section 7. Predict One New Case ===


def predict_example(model: DecisionTreeClassifier) -> None:
    """Use the trained model to predict pass/fail for one new student."""
    LOG.info("Predicting one new case")

    new_case = pd.DataFrame(
        [
            {
                "hours_studied": 6.5,
                "practice_quizzes": 4,
                "attendance_pct": 92,
                "sleep_hours": 7.0,
                "prior_score": 72,
            }
        ]
    )

    predicted_label: str = model.predict(new_case)[0]

    LOG.info(f"New case:\n{new_case}")
    LOG.info(f"Predicted result: {predicted_label}")


# === Section 8. Create Visualizations ===


def make_plots(df_clean: pd.DataFrame, model: DecisionTreeClassifier) -> None:
    """Create charts for the supervised classification case."""
    LOG.info("Creating chart: hours studied vs prior score by result")

    fig, ax = plt.subplots(figsize=(9, 5))

    scatter_plt: Axes = sns.scatterplot(
        data=df_clean,
        x="hours_studied",
        y="prior_score",
        hue=TARGET_COL,
        ax=ax,
    )

    scatter_plt.set_title(
        "Hours Studied vs Prior Score by Result (CLOSE chart to continue)"
    )
    scatter_plt.set_xlabel("Hours Studied")
    scatter_plt.set_ylabel("Prior Score")

    LOG.info("Creating chart: confusion matrix heatmap")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]
    y_pred = model.predict(x)

    labels: list[str] = ["fail", "pass"]
    matrix = confusion_matrix(y, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))

    heatmap_plt: Axes = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    heatmap_plt.set_title("Confusion Matrix - All Rows (CLOSE chart to continue)")
    heatmap_plt.set_xlabel("Predicted")
    heatmap_plt.set_ylabel("True")


# === Section 9. Summary and Next Steps ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows: {df_clean.shape[0]}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Target: {TARGET_COL} (pass = score >= {PASS_THRESHOLD})")
    LOG.info(f"Class balance:\n{df_clean[TARGET_COL].value_counts()}")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the supervised ML workflow."""
    log_header(LOG, "ML")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Derive target.............")
    df = derive_target(df)

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view.........")
    df_clean = make_clean_view(df)

    LOG.info("Train supervised model....")
    model = train_model(df_clean)

    LOG.info("Predict one case..........")
    predict_example(model)

    LOG.info("Create charts.............")
    make_plots(df_clean, model)

    LOG.info("Summarize workflow........")
    summarize(df, df_clean)

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
