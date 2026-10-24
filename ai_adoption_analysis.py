"""Reproducible descriptive analysis for the AI-on-society survey.

Research plan
-------------
1. Knowledge: Is self-reported AI knowledge associated with AI usage and
   willingness to adopt?
2. Trust: Is lower trust associated with lower willingness to use more AI?
3. Perceived risks: Are job-displacement and individual-freedom concerns
   associated with lower willingness to adopt?
4. Demographics: Do these patterns differ by age, education, or employment?

The survey contains no open-ended responses, so this script summarizes
predefined survey themes rather than claiming to perform qualitative coding.
Ordinal scores are for exploratory analysis only; they do not imply equal
distances between response options.

Usage:
    python ai_adoption_analysis.py
    python ai_adoption_analysis.py --data path/to/AI_on_society.csv
    python ai_adoption_analysis.py --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = {
    "age_group": "What is your age range?",
    "gender": "What is your gender?",
    "education_level": "What is your education level?",
    "employment_status": "What is your employment status?",
    "occupation": "What is your occupation? (optional)",
    "device_use": "How often do you use technological devices?",
    "ai_knowledge": (
        "How much knowledge do you have about artificial intelligence "
        "(AI) technologies?"
    ),
    "ai_trust": "Do you generally trust artificial intelligence (AI)?",
    "ai_humanity_impact": (
        "Do you think artificial intelligence (AI) will be generally "
        "beneficial or harmful to humanity?"
    ),
    "ai_usage": (
        "Please rate how actively you use AI-powered products in your daily "
        "life on a scale from 1 to 5."
    ),
    "future_willingness": "Would you like to use more AI products in the future?",
    "freedom_concern": (
        "I think artificial intelligence (AI) could threaten individual freedoms."
    ),
    "job_elimination_belief": (
        "Could artificial intelligence (AI) completely eliminate some professions?"
    ),
    "own_job_impact": (
        "Do you think your own job could be affected by artificial "
        "intelligence (AI)?"
    ),
    "ethical_rules": (
        "Do you believe that artificial intelligence (AI) should be limited "
        "by ethical rules?"
    ),
    "ai_consciousness_belief": (
        "Could artificial intelligence (AI) one day become conscious like humans?"
    ),
    "ai_application_knowledge": (
        "Which of the following do you think is NOT an artificial "
        "intelligence (AI) application?"
    ),
    "machine_learning_knowledge": (
        "Which of the following is a machine learning algorithm used in the "
        "field of artificial intelligence?"
    ),
    "chatgpt_knowledge": (
        "The artificial intelligence application called 'ChatGPT' is an "
        "example of which type of AI system?"
    ),
}

ORDINAL_MAPPINGS = {
    "knowledge_score": (
        "ai_knowledge",
        {
            "I have no knowledge": 1,
            "I've heard a little about it": 2,
            "I have basic knowledge": 3,
            "I have a good level of knowledge": 4,
            "I have an expert-level knowledge": 5,
        },
    ),
    "trust_score": (
        "ai_trust",
        {
            "I don't trust it at all": 1,
            "I don't trust it": 2,
            "I'm undecided": 3,
            "I trust it": 4,
            "I fully trust it": 5,
        },
    ),
    "willingness_score": (
        "future_willingness",
        {
            "Definitely, I would not like to": 1,
            "I would not like to": 2,
            "I'm undecided": 3,
            "I would like to": 4,
            "Definitely, I would like to": 5,
        },
    ),
    "freedom_concern_score": (
        "freedom_concern",
        {
            "Strongly disagree": 1,
            "I disagree": 2,
            "I'm undecided": 3,
            "Agree": 4,
            "Strongly Agree": 5,
        },
    ),
    "job_elimination_belief_score": (
        "job_elimination_belief",
        {
            "Absolutely Can't handle it": 1,
            "Can't handle it": 2,
            "I have no idea": 3,
            "Removes": 4,
            "Definitely Removes": 5,
        },
    ),
    "own_job_impact_score": (
        "own_job_impact",
        {
            "Definitely I don't think so": 1,
            "I don't think so": 2,
            "I'm undecided": 3,
            "Think": 4,
            "I definitely think": 5,
        },
    ),
    "ethical_rules_score": (
        "ethical_rules",
        {
            "Strongly disagree": 1,
            "I disagree": 2,
            "I'm undecided": 3,
            "Agree": 4,
            "Strongly Agree": 5,
        },
    ),
}

DEMOGRAPHICS = ["age_group", "education_level", "employment_status"]
KEY_CATEGORICALS = [
    "age_group",
    "gender",
    "education_level",
    "employment_status",
    "device_use",
    "ai_knowledge",
    "ai_trust",
    "ai_usage",
    "future_willingness",
    "freedom_concern",
    "job_elimination_belief",
    "own_job_impact",
    "ethical_rules",
    "ai_humanity_impact",
]
KEY_SCORES = [
    "knowledge_score",
    "trust_score",
    "ai_usage",
    "willingness_score",
    "freedom_concern_score",
    "job_elimination_belief_score",
    "own_job_impact_score",
    "ethical_rules_score",
]


def load_data(path: Path) -> pd.DataFrame:
    """Read the source CSV and standardize labels needed for analysis."""
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="cp1252")

    missing_columns = sorted(set(SOURCE_COLUMNS.values()) - set(raw.columns))
    if missing_columns:
        raise ValueError(
            "The source CSV is missing expected survey columns: "
            + ", ".join(missing_columns)
        )

    data = raw.drop(columns=["ID"], errors="ignore").rename(columns=SOURCE_COLUMNS)
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].str.strip()

    data["education_level"] = data["education_level"].replace(
        {"Ön Bachelor's degree": "Associate degree"}
    )
    data["employment_status"] = data["employment_status"].replace(
        {"Emekliyim": "Retired"}
    )

    for score, (source, mapping) in ORDINAL_MAPPINGS.items():
        data[score] = data[source].map(mapping)
        unmapped = data.loc[data[source].notna() & data[score].isna(), source]
        if not unmapped.empty:
            labels = ", ".join(sorted(unmapped.astype(str).unique()))
            raise ValueError(
                f"Unmapped responses for {source} ({score}): {labels}. "
                "Review the coding map before interpreting results."
            )

    data["ai_usage"] = pd.to_numeric(data["ai_usage"], errors="coerce")
    return data


def frequency_table(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    tables = []
    for column in columns:
        counts = data[column].value_counts(dropna=False, sort=False)
        table = counts.rename("count").to_frame()
        table["percent"] = (counts / len(data) * 100).round(1)
        table.insert(0, "variable", column)
        table.insert(1, "response", table.index.map(
            lambda value: "(missing)" if pd.isna(value) else str(value)
        ))
        tables.append(table.reset_index(drop=True))
    return pd.concat(tables, ignore_index=True)


def numeric_summary(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = data[column].dropna()
        rows.append(
            {
                "variable": column,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows).round(2)


def subgroup_summary(data: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["ai_usage", "willingness_score"]
    rows = []
    for demographic in DEMOGRAPHICS:
        for group, subset in data.groupby(demographic, dropna=False, observed=False):
            for outcome in outcomes:
                values = subset[outcome].dropna()
                rows.append(
                    {
                        "demographic": demographic,
                        "group": "(missing)" if pd.isna(group) else group,
                        "outcome": outcome,
                        "n": len(values),
                        "mean": values.mean(),
                        "median": values.median(),
                    }
                )
    return pd.DataFrame(rows).round(2)


def run_analysis(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frequencies = frequency_table(data, KEY_CATEGORICALS)
    score_summary = numeric_summary(data, KEY_SCORES)
    subgroup = subgroup_summary(data)
    correlations = data[KEY_SCORES].corr(method="spearman").round(3)
    missingness = (
        data.isna()
        .sum()
        .rename("missing_n")
        .to_frame()
        .assign(missing_percent=lambda frame: (frame["missing_n"] / len(data) * 100).round(1))
        .reset_index(names="variable")
    )

    frequencies.to_csv(output_dir / "response_frequencies.csv", index=False)
    score_summary.to_csv(output_dir / "ordinal_score_descriptives.csv", index=False)
    subgroup.to_csv(output_dir / "demographic_subgroups.csv", index=False)
    correlations.to_csv(output_dir / "spearman_correlations.csv")
    missingness.to_csv(output_dir / "missing_data.csv", index=False)

    print(f"Respondents: {len(data)}")
    print(f"Results written to: {output_dir.resolve()}")
    print("\nOrdinal-score descriptives (exploratory):")
    print(score_summary.to_string(index=False))
    print("\nSpearman correlations (exploratory associations, not causal effects):")
    print(correlations.to_string())
    print("\nDemographic subgroup means for use and willingness:")
    print(subgroup.to_string(index=False))
    print("\nMissing data:")
    print(missingness.loc[missingness["missing_n"] > 0].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("AI_on_society.csv"),
        help="Path to the raw survey CSV (default: AI_on_society.csv beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output"),
        help="Directory for generated CSV tables (default: ./analysis_output).",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(
            f"Survey CSV not found at {args.data}. Download AI_on_society.csv "
            "from the repository or provide its path with --data."
        )

    run_analysis(load_data(args.data), args.output_dir)


if __name__ == "__main__":
    main()
