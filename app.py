"""Interactive, aggregate-only dashboard for the AI adoption survey."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_adoption_analysis import (
    KEY_CATEGORICALS,
    KEY_SCORES,
    load_data,
)


st.set_page_config(
    page_title="Understanding AI Adoption",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data(Path(__file__).with_name("AI_on_society.csv"))


def share(values: pd.Series, predicate) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    return float(predicate(values).mean() * 100)


def response_distribution(data: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = data[column].value_counts(dropna=False)
    result = counts.rename("Respondents").to_frame()
    result["Percent"] = (counts / len(data) * 100).round(1)
    result.index = result.index.map(
        lambda value: "(Missing)" if pd.isna(value) else str(value)
    )
    result.index.name = "Response"
    return result


def relationship_summary(
    data: pd.DataFrame, factor: str, factor_label: str
) -> tuple[pd.DataFrame, float]:
    summary = (
        data.groupby(factor, observed=True)
        .agg(
            respondents=("willingness_score", "count"),
            mean_willingness=("willingness_score", "mean"),
            median_willingness=("willingness_score", "median"),
            mean_usage=("ai_usage", "mean"),
        )
        .round(2)
        .reset_index()
        .rename(
            columns={
                factor: factor_label,
                "respondents": "Respondents",
                "mean_willingness": "Mean willingness (1-5)",
                "median_willingness": "Median willingness (1-5)",
                "mean_usage": "Mean AI use (1-5)",
            }
        )
    )
    rho = data[[factor, "willingness_score"]].corr(method="spearman").iloc[0, 1]
    return summary, float(rho)


try:
    survey = get_data()
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load or validate the survey dataset: {error}")
    st.stop()

st.title("Understanding AI Adoption")
st.caption(
    "An exploratory look at AI knowledge, trust, perceived risks, usage, and "
    "future willingness."
)
st.info(
    "This public dashboard displays aggregate results only, not individual "
    "respondent records. The survey is cross-sectional and should not be used "
    "to infer causation or population-wide estimates."
)

with st.sidebar:
    st.header("Explore the sample")
    selections = {}
    for column, label in [
        ("age_group", "Age group"),
        ("education_level", "Education"),
        ("employment_status", "Employment"),
    ]:
        options = sorted(survey[column].dropna().unique().tolist())
        selections[column] = st.multiselect(label, options, default=options)
    st.caption("Choose respondent groups to include in the analysis.")

filtered = survey.copy()
for column, selected in selections.items():
    if selected:
        filtered = filtered[filtered[column].isin(selected)]
    else:
        filtered = filtered.iloc[0:0]

if filtered.empty:
    st.warning(
        "No respondents match these filters. Select at least one group per filter."
    )
    st.stop()

total_respondents = len(filtered)
under_18 = int((filtered["age_group"] == "Under 18 years old").sum())
want_more = share(filtered["willingness_score"], lambda values: values >= 4)
low_trust = share(filtered["trust_score"], lambda values: values <= 2)
mean_usage = filtered["ai_usage"].mean()

metric_columns = st.columns(4)
metric_columns[0].metric("Respondents", f"{total_respondents:,}")
metric_columns[1].metric("Would like to use more AI", f"{want_more:.1f}%")
metric_columns[2].metric("Distrust AI", f"{low_trust:.1f}%")
metric_columns[3].metric(
    "Mean AI product use", f"{mean_usage:.2f}/5" if pd.notna(mean_usage) else "N/A"
)

if under_18:
    st.caption(
        f"{under_18:,} respondents in the current selection are under 18. "
        "Interpret these results with appropriate care."
    )

knowledge, knowledge_rho = relationship_summary(
    filtered, "knowledge_score", "Self-reported AI knowledge (1-5)"
)
trust, trust_rho = relationship_summary(
    filtered, "trust_score", "Trust in AI (1-5)"
)
job_impact, job_impact_rho = relationship_summary(
    filtered, "own_job_impact_score", "Perceived impact on own job (1-5)"
)

summary_tab, overview_tab, knowledge_tab, risk_tab, demographic_tab, details_tab = st.tabs(
    [
        "Summary of findings",
        "Overview & themes",
        "Knowledge and trust",
        "Perceived risks",
        "Demographics",
        "Descriptives",
    ]
)

with summary_tab:
    st.subheader("Summary of findings")
    st.markdown(
        "This survey suggests that willingness to adopt AI is shaped more by trust than by knowledge alone. "
        "In descriptive terms, the relationship between trust and willingness appears stronger than the relationship between "
        "self-reported AI knowledge and willingness. This indicates that respondents may need more than awareness of AI tools; "
        "they also need confidence that AI is trustworthy, safe, and appropriate for everyday use."
    )
    st.markdown(
        "The positive association between perceived job impact and willingness is also notable. In a simple risk-based interpretation, "
        "one might expect people who think AI will affect their job to be more resistant to adoption. Instead, the pattern in this sample is the opposite. "
        "A plausible explanation is that people who are more aware of AI's labour implications are also more engaged with AI and therefore more open to adapting to it, but this remains a descriptive pattern rather than a causal finding."
    )
    st.markdown(
        "The survey also shows that uncertainty is a central theme: a meaningful share of respondents are undecided about trust, illustrating that "
        "hesitation may be as important as scepticism. Overall, these results support a cautious interpretation. The sample is modest and cross-sectional, and self-reported attitudes cannot be used to draw firm policy conclusions or predict broader population behaviour."
    )

    highlight_cols = st.columns(3)
    highlight_cols[0].metric("Trust-willingness association", f"ρ ≈ {trust_rho:.2f}")
    highlight_cols[1].metric("Knowledge-willingness association", f"ρ ≈ {knowledge_rho:.2f}")
    highlight_cols[2].metric("Job impact-willingness association", f"ρ ≈ {job_impact_rho:.2f}")

    st.caption(
        "Interpretation note: these are exploratory Spearman associations, not causal estimates. "
        "Policy recommendations should be treated as tentative until validated in a larger and more representative study."
    )

with overview_tab:
    st.subheader("What stands out in this sample?")
    themes = st.columns(4)
    themes[0].markdown("**Knowledge gap**")
    themes[0].write(
        f"{share(filtered['knowledge_score'], lambda values: values <= 3):.1f}% "
        "report basic knowledge or less."
    )
    themes[1].markdown("**Trust uncertainty**")
    themes[1].write(
        f"{share(filtered['trust_score'], lambda values: values == 3):.1f}% "
        "are undecided about trusting AI."
    )
    themes[2].markdown("**Adoption openness**")
    themes[2].write(f"{want_more:.1f}% say they would like to use more AI.")
    themes[3].markdown("**Risk awareness**")
    themes[3].write(
        f"{share(filtered['freedom_concern_score'], lambda values: values >= 4):.1f}% "
        "agree AI could threaten individual freedoms."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Current AI product use**")
        st.bar_chart(response_distribution(filtered, "ai_usage")["Respondents"])
        st.caption("Self-rated daily use (1 = least active, 5 = most active).")
    with right:
        st.markdown("**Willingness to use more AI**")
        st.bar_chart(
            response_distribution(filtered, "future_willingness")["Respondents"]
        )
        st.caption("Response counts for future adoption willingness.")

with knowledge_tab:
    st.subheader("Knowledge and trust")
    left, right = st.columns(2)
    with left:
        st.markdown("**Knowledge and adoption**")
        st.caption(
            f"Spearman association with willingness: ρ = {knowledge_rho:.2f}. "
            "This is descriptive, not causal."
        )
        st.bar_chart(
            knowledge.set_index("Self-reported AI knowledge (1-5)")[
                ["Mean willingness (1-5)", "Mean AI use (1-5)"]
            ]
        )
        st.dataframe(knowledge, hide_index=True, width="stretch")
    with right:
        st.markdown("**Trust and adoption**")
        st.caption(
            f"Spearman association with willingness: ρ = {trust_rho:.2f}. "
            "This is descriptive, not causal."
        )
        st.bar_chart(
            trust.set_index("Trust in AI (1-5)")[
                ["Mean willingness (1-5)", "Mean AI use (1-5)"]
            ]
        )
        st.dataframe(trust, hide_index=True, width="stretch")

with risk_tab:
    st.subheader("Perceived risks and adoption willingness")
    risk_options = [
        ("freedom_concern_score", "Concern about individual freedoms (1-5)"),
        ("job_elimination_belief_score", "Belief jobs may be eliminated (1-5)"),
        ("own_job_impact_score", "Perceived impact on own job (1-5)"),
    ]
    for start in range(0, len(risk_options), 2):
        columns = st.columns(2)
        for slot, (score_column, label) in enumerate(risk_options[start : start + 2]):
            summary, rho = relationship_summary(filtered, score_column, label)
            with columns[slot]:
                st.markdown(f"**{label}**")
                st.caption(
                    f"Spearman association with willingness: ρ = {rho:.2f}. "
                    "Interpret ordinal trends cautiously."
                )
                st.bar_chart(
                    summary.set_index(label)[
                        ["Mean willingness (1-5)", "Mean AI use (1-5)"]
                    ]
                )
                st.dataframe(summary, hide_index=True, width="stretch")

with demographic_tab:
    st.subheader("How results vary across demographic groups")
    for column, label in [
        ("age_group", "Age group"),
        ("education_level", "Education level"),
        ("employment_status", "Employment status"),
    ]:
        grouped = (
            filtered.groupby(column, observed=True)
            .agg(
                respondents=("willingness_score", "count"),
                mean_usage=("ai_usage", "mean"),
                mean_willingness=("willingness_score", "mean"),
            )
            .round(2)
        )
        st.markdown(f"**By {label.lower()}**")
        st.bar_chart(
            grouped.rename(
                columns={
                    "mean_usage": "Mean AI use (1-5)",
                    "mean_willingness": "Mean willingness (1-5)",
                }
            )[["Mean AI use (1-5)", "Mean willingness (1-5)"]]
        )
        st.dataframe(
            grouped.rename(
                columns={
                    "respondents": "Respondents",
                    "mean_usage": "Mean AI use (1-5)",
                    "mean_willingness": "Mean willingness (1-5)",
                }
            ),
            width="stretch",
        )

with details_tab:
    st.subheader("Descriptive statistics")
    st.markdown("**Response distributions**")
    selected_variable = st.selectbox(
        "Survey item",
        KEY_CATEGORICALS,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    distribution = response_distribution(filtered, selected_variable)
    st.dataframe(distribution, width="stretch")
    st.markdown("**Ordinal-score summaries**")
    score_summary = (
        filtered[KEY_SCORES]
        .describe()
        .T[["count", "mean", "std", "50%", "min", "max"]]
        .rename(
            columns={
                "count": "N",
                "mean": "Mean",
                "std": "SD",
                "50%": "Median",
                "min": "Min",
                "max": "Max",
            }
        )
        .round(2)
    )
    st.dataframe(score_summary, width="stretch")
    st.markdown("**Missing responses**")
    missing = (
        filtered.isna()
        .sum()
        .rename("Missing")
        .to_frame()
        .assign(
            Percent=lambda frame: (
                frame["Missing"] / total_respondents * 100
            ).round(1)
        )
        .query("Missing > 0")
    )
    if missing.empty:
        st.write("No missing responses in the selected sample.")
    else:
        st.dataframe(missing, width="stretch")

st.divider()
st.caption(
    "Source: AI_on_society.csv in the oyecrafts/Understanding-AI-Adoption "
    "repository. Ordinal scores encode ordered response categories for "
    "exploration; intervals between categories are not assumed equal."
)
