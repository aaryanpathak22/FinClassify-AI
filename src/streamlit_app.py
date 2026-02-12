import streamlit as st
import pandas as pd
from inference_service import FinClassifyService


@st.cache_resource
def get_service() -> FinClassifyService:
    return FinClassifyService()


def main() -> None:
    st.set_page_config(page_title="FinClassify AI",  layout="centered")
    st.title("FinClassify AI – Smart Transaction Categorisation")
    st.write(
        "Paste any **raw transaction description** (e.g. "
        "`Starbud Gas #2345 NY`, `UBER *TRIP 8877`, `DMART GROCERY BILL`) "
        "and I'll classify it into a category with an explanation."
    )

    service = get_service()

    text = st.text_input(
        "Transaction text",
        value="AMAZON MKTPLACE PMT 8877",
        help="Copy this from a bank statement, SMS, or transaction log.",
    )

    if st.button("Classify transaction", type="primary"):
        if not text.strip():
            st.warning("Please enter a transaction description.")
            return

        with st.spinner("Running FinClassify AI..."):
            pred = service.predict(text)
            exps = service.explain(text)

        st.subheader("Prediction")
        st.markdown(
            f"**Category:** `{pred['predicted_category']}`  \n"
            f"**Confidence:** `{pred['confidence']:.2f}`"
        )

        # ✅ FIXED CLASS SCORES BAR CHART
        st.subheader("Class scores")

        scores = pred["scores"]
        df_scores = pd.DataFrame({
            "Category": list(scores.keys()),
            "Score": list(scores.values())
        })

        # Optional: sort by score (highest first)
        df_scores = df_scores.sort_values(by="Score", ascending=False)

        st.bar_chart(df_scores, x="Category", y="Score")

        # ✅ EXPLANATION SECTION
        st.subheader("Why this category? (simple XAI)")
        if exps:
            for item in exps:
                st.write(
                    f"- Token **`{item['token']}`** reduced confidence by "
                    f"`{item['delta_confidence']:.3f}` when removed."
                )
        else:
            st.write("_Not enough tokens for explanation._")

        # ✅ FEEDBACK SECTION
        st.subheader("Feedback (continuous learning)")
        st.write(
            "Help improve the model by telling us if the prediction was correct. "
            "Your feedback is logged for future retraining."
        )

        is_correct = st.radio(
            "Is this category correct?",
            options=["Yes", "No"],
            index=0,
            horizontal=True,
        )

        if is_correct == "No":
            all_labels = service.class_names
            default_idx = (
                all_labels.index(pred["predicted_category"])
                if pred["predicted_category"] in all_labels
                else 0
            )
            correct_cat = st.selectbox(
                "Select the correct category:",
                options=all_labels,
                index=default_idx,
            )

            if st.button("Submit feedback"):
                service.log_feedback(
                    raw_text=text,
                    predicted_category=pred["predicted_category"],
                    correct_category=correct_cat,
                )
                st.success("Thanks! Your feedback has been recorded.")
        else:
            st.info("Great! No feedback change recorded.")


if __name__ == "__main__":
    main()
