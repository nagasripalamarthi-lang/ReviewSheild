"""ReviewShield – Fake Review Detection System (Streamlit Application)."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    BULK_SAMPLE_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    LABEL_GENUINE,
    LABEL_SUSPICIOUS,
    SAMPLE_DATASET_PATH,
    SUSPICIOUS_THRESHOLD,
)
from src.database import (
    clear_analysis_history,
    init_db,
    load_analysis_history,
    save_analysis_record,
    save_bulk_records,
)
from src.duplicate_detector import analyze_duplicates_and_patterns
from src.ml_model import (
    load_and_validate_dataset,
    load_or_train_model,
    predict_batch,
    predict_single_review,
    train_and_save_model,
)
from src.nlp_preprocessing import STYLOMETRIC_DISPLAY_NAMES
from src.ui_components import (
    inject_custom_css,
    render_confidence_gauge,
    render_confidence_histogram,
    render_confusion_matrix_figure,
    render_disclaimer_banner,
    render_distribution_donut,
    render_feature_drivers_chart,
    render_hero,
    render_kpi_card,
)

st.set_page_config(
    page_title="ReviewShield – Fake Review Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading ReviewShield NLP & ML model...")
def get_model_bundle():
    """Load or train the ReviewShield scikit-learn pipeline once per session."""
    return load_or_train_model()


def _detect_text_column(columns: list[str]) -> str:
    """Heuristically pick the most likely review text column from a CSV header."""
    preferred = ["review_text", "review", "text", "comment", "content", "body", "feedback"]
    lowered = {col.lower().strip(): col for col in columns}
    for cand in preferred:
        if cand in lowered:
            return lowered[cand]
    return columns[0]


def _render_dashboard_page(pipeline, metadata: dict) -> None:
    render_hero(
        "🛡️ ReviewShield – Review Insights Dashboard",
        "Monitor analyzed product reviews, suspicion rates, score distributions, and persistent SQLite audit logs.",
    )
    render_disclaimer_banner()

    history_df = load_analysis_history(limit=1000)
    active_bulk: pd.DataFrame | None = st.session_state.get("bulk_results")

    source_options = ["SQLite Saved History"]
    if active_bulk is not None and not active_bulk.empty:
        source_options.append("Current Bulk Session")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1.4])
    with col_ctrl1:
        data_source = st.radio(
            "Dashboard Data Source",
            options=source_options,
            horizontal=True,
        )
    with col_ctrl2:
        if st.button("Populate Dashboard with Sample Bulk Batch", width="stretch"):
            sample_bulk = pd.read_csv(BULK_SAMPLE_PATH)
            analyzed_sample = predict_batch(sample_bulk, text_column="review_text", pipeline=pipeline)
            save_bulk_records(analyzed_sample, source="sample_bulk_demo")
            st.session_state["bulk_results"] = analyzed_sample
            st.success(f"Analyzed and saved {len(analyzed_sample)} sample reviews to SQLite history.")
            st.rerun()
    with col_ctrl3:
        if not history_df.empty and st.button("Clear SQLite History", width="stretch"):
            clear_analysis_history()
            st.info("Cleared saved SQLite analysis history.")
            st.rerun()

    df = active_bulk if (data_source == "Current Bulk Session" and active_bulk is not None) else history_df

    if df.empty:
        st.info(
            "No reviews have been saved to the dashboard yet. Use **Single Review Analysis**, "
            "**Bulk CSV Detection**, or click **Populate Dashboard with Sample Bulk Batch** above."
        )
        return

    total_reviews = len(df)
    suspicious_count = int((df["prediction"] == LABEL_SUSPICIOUS).sum())
    genuine_count = int((df["prediction"] == LABEL_GENUINE).sum())
    suspicious_pct = (suspicious_count / total_reviews * 100.0) if total_reviews else 0.0
    avg_conf = float(df["confidence"].mean() * 100.0) if total_reviews else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total Reviews Analyzed", f"{total_reviews:,}", f"Source: {data_source}")
    with k2:
        render_kpi_card(
            "Potentially Suspicious",
            f"{suspicious_count:,}",
            f"{suspicious_pct:.1f}% of analyzed reviews",
            accent="#DC2626",
        )
    with k3:
        render_kpi_card(
            "Potentially Genuine",
            f"{genuine_count:,}",
            f"{100.0 - suspicious_pct:.1f}% of analyzed reviews",
            accent="#16A34A",
        )
    with k4:
        render_kpi_card(
            "Avg Model Confidence",
            f"{avg_conf:.1f}%",
            "Mean predicted class probability",
            accent="#2563EB",
        )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(render_distribution_donut(genuine_count, suspicious_count), width="stretch")
    with chart_col2:
        st.plotly_chart(render_confidence_histogram(df), width="stretch")

    # Additional behavioral insights: word count vs suspicion probability
    insight_df = df.copy()
    insight_df["word_count"] = insight_df["review_text"].astype(str).str.split().str.len()
    insight_df["exclamation_count"] = insight_df["review_text"].astype(str).str.count("!")

    c_left, c_right = st.columns(2)
    with c_left:
        fig_len = px.box(
            insight_df,
            x="prediction",
            y="word_count",
            color="prediction",
            color_discrete_map={LABEL_GENUINE: "#22C55E", LABEL_SUSPICIOUS: "#EF4444"},
            title="Review Word Count by Classification",
            labels={"prediction": "Classification", "word_count": "Word Count"},
        )
        fig_len.update_layout(height=320, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC")
        st.plotly_chart(fig_len, width="stretch")

    with c_right:
        fig_excl = px.scatter(
            insight_df,
            x="word_count",
            y="suspicious_probability",
            color="prediction",
            size=insight_df["exclamation_count"].clip(lower=0) + 1,
            hover_data=["product_name"] if "product_name" in insight_df.columns else None,
            color_discrete_map={LABEL_GENUINE: "#22C55E", LABEL_SUSPICIOUS: "#EF4444"},
            title="Word Count vs. Suspicion Score (bubble size = '!' count)",
            labels={"word_count": "Word Count", "suspicious_probability": "Suspicion Probability"},
        )
        fig_excl.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC")
        st.plotly_chart(fig_excl, width="stretch")

    st.subheader("Recent Analyzed Reviews")
    display_cols = [
        col
        for col in ["created_at", "product_name", "rating", "prediction", "confidence", "review_text", "explanation_summary"]
        if col in df.columns
    ]
    preview = df[display_cols].copy()
    if "confidence" in preview.columns:
        preview["confidence"] = (preview["confidence"] * 100.0).round(1).astype(str) + "%"
    st.dataframe(preview, width="stretch", height=320)


def _render_single_review_page(pipeline) -> None:
    render_hero(
        "🔍 Single Review Analysis",
        "Inspect an individual product review for linguistic risk signals, model confidence, and feature-level explanations.",
    )
    render_disclaimer_banner()

    presets = {
        "Balanced Genuine Review": (
            "AeroSound Wireless Earbuds",
            4,
            "Battery lasts about 6 hours per charge and the USB-C case tops it up twice. "
            "Passive noise isolation is decent on my commute, though the touch controls are slightly sensitive.",
        ),
        "Promotional / Incentivized Spam": (
            "VoltCharge 65W GaN Adapter",
            5,
            "BEST CHARGER EVER!!!! 100% GENUINE MUST BUY RIGHT NOW!!!! "
            "DM me on telegram for promo code and full cashback refund after 5 stars!!",
        ),
        "Honest Negative Review (Genuine)": (
            "Lumina 27-inch Monitor",
            2,
            "Backlight bleed is noticeable in the bottom-left corner during dark movies. "
            "Fine for spreadsheets, disappointing for evening gaming. Returned within 14 days.",
        ),
        "Competitor-Bashing Spam": (
            "PureAir HEPA Purifier",
            1,
            "TOTAL SCAM GARBAGE WORST PRODUCT EVER DO NOT BUY!!!! CLICK LINK www.example-deals.test",
        ),
    }

    st.markdown("**Quick Test Examples:**")
    p_cols = st.columns(len(presets))
    for col, (label, (prod, stars, txt)) in zip(p_cols, presets.items()):
        with col:
            if st.button(label, width="stretch"):
                st.session_state["single_product"] = prod
                st.session_state["single_rating"] = stars
                st.session_state["single_text"] = txt

    with st.form("single_review_form"):
        f1, f2 = st.columns([3, 1])
        with f1:
            product_name = st.text_input(
                "Product Name (optional)",
                value=st.session_state.get("single_product", ""),
                placeholder="e.g., AeroSound Wireless Earbuds",
            )
        with f2:
            rating = st.selectbox(
                "Star Rating (optional)",
                options=[5, 4, 3, 2, 1],
                index=[5, 4, 3, 2, 1].index(st.session_state.get("single_rating", 5)),
            )

        review_text = st.text_area(
            "Product Review Text",
            value=st.session_state.get("single_text", ""),
            height=140,
            placeholder="Paste or type a product review here to analyze...",
        )

        opt1, opt2 = st.columns([2, 1])
        with opt1:
            threshold = st.slider(
                "Suspicion Decision Threshold",
                min_value=0.30,
                max_value=0.80,
                value=SUSPICIOUS_THRESHOLD,
                step=0.05,
                help="Reviews with predicted suspicion probability at or above this threshold are flagged Potentially Suspicious.",
            )
        with opt2:
            save_to_db = st.checkbox("Save result to SQLite history", value=True)

        submitted = st.form_submit_button("Analyze Review", type="primary", width="stretch")

    if submitted:
        st.session_state["single_product"] = product_name
        st.session_state["single_rating"] = rating
        st.session_state["single_text"] = review_text

        if not review_text.strip():
            st.warning("Please enter a product review before clicking Analyze Review.")
            return

        result = predict_single_review(review_text, pipeline=pipeline, threshold=threshold)
        if save_to_db:
            save_analysis_record(
                review_text=review_text.strip(),
                prediction=result["prediction"],
                suspicious_probability=result["suspicious_probability"],
                confidence=result["confidence"],
                explanation_summary=result["explanation_summary"],
                signals=result["signals"],
                source="single_analysis",
                product_name=product_name,
                rating=float(rating),
            )

        st.divider()
        top_left, top_right = st.columns([1.3, 1])
        with top_left:
            is_suspicious = result["prediction"] == LABEL_SUSPICIOUS
            badge_cls = "rs-badge-suspicious" if is_suspicious else "rs-badge-genuine"
            st.markdown(f'<span class="{badge_cls}">{escape(result["prediction"])}</span>', unsafe_allow_html=True)
            st.markdown(f"### Confidence: **{result['confidence'] * 100:.1f}%**")
            st.write(result["explanation_summary"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Suspicious Prob.", f"{result['suspicious_probability'] * 100:.1f}%")
            m2.metric("Genuine Prob.", f"{result['genuine_probability'] * 100:.1f}%")
            m3.metric("Word Count", int(result["stylometric_features"]["word_count"]))

        with top_right:
            st.plotly_chart(
                render_confidence_gauge(result["suspicious_probability"], result["prediction"]),
                width="stretch",
            )

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.subheader("Why It Got This Classification")
            for sig in result["signals"]:
                sev_class = f"rs-signal-{sig['severity']}"
                st.markdown(
                    f"""
                    <div class="rs-signal-card {sev_class}">
                        <strong>{escape(sig['title'])}</strong><br/>
                        <span style="font-size:0.9rem;color:#334155;">{escape(sig['detail'])}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with exp_col2:
            st.subheader("Top Model Token & Feature Drivers")
            fig_drivers = render_feature_drivers_chart(
                result["top_suspicious_drivers"],
                result["top_genuine_drivers"],
            )
            if fig_drivers is not None:
                st.plotly_chart(fig_drivers, width="stretch")
            else:
                st.info("No strong linear feature drivers exceeded the display threshold.")

        with st.expander("View Full Stylometric Feature Vector"):
            stylo_rows = [
                {
                    "Feature": STYLOMETRIC_DISPLAY_NAMES.get(k, k),
                    "Value": round(v, 3),
                }
                for k, v in result["stylometric_features"].items()
            ]
            st.dataframe(pd.DataFrame(stylo_rows), width="stretch", hide_index=True)


def _render_bulk_page(pipeline) -> None:
    render_hero(
        "📂 Bulk CSV Review Detection",
        "Upload a CSV dataset of product reviews to score every row, separate suspicious entries, and export results.",
    )
    render_disclaimer_banner()

    c1, c2 = st.columns([2.2, 1])
    with c1:
        uploaded = st.file_uploader(
            "Upload Review CSV File",
            type=["csv"],
            help="CSV should contain a review text column (e.g., review_text, review, text, comment).",
        )
    with c2:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        use_sample = st.button("Load Sample Bulk CSV (30 Reviews)", width="stretch")

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            st.session_state["bulk_raw_df"] = raw_df
            st.session_state["bulk_source_name"] = uploaded.name
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
            return
    elif use_sample:
        raw_df = pd.read_csv(BULK_SAMPLE_PATH)
        st.session_state["bulk_raw_df"] = raw_df
        st.session_state["bulk_source_name"] = BULK_SAMPLE_PATH.name

    raw_df = st.session_state.get("bulk_raw_df")
    if raw_df is None or raw_df.empty:
        st.info("Upload a CSV file or click **Load Sample Bulk CSV (30 Reviews)** to begin batch detection.")
        return

    st.caption(f"Loaded **{len(raw_df):,} rows** from `{st.session_state.get('bulk_source_name', 'CSV')}`.")

    col_sel, col_thresh, col_btn = st.columns([1.8, 1.5, 1.2])
    with col_sel:
        default_col = _detect_text_column(raw_df.columns.tolist())
        text_column = st.selectbox(
            "Select Review Text Column",
            options=raw_df.columns.tolist(),
            index=raw_df.columns.tolist().index(default_col),
        )
    with col_thresh:
        threshold = st.slider("Suspicion Threshold", 0.30, 0.80, SUSPICIOUS_THRESHOLD, 0.05)
    with col_btn:
        st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
        run_bulk = st.button("Analyze All Reviews", type="primary", width="stretch")

    if run_bulk:
        with st.spinner("Running NLP preprocessing & ML classification..."):
            analyzed_df = predict_batch(raw_df, text_column=text_column, pipeline=pipeline, threshold=threshold)
        st.session_state["bulk_results"] = analyzed_df

    analyzed_df: pd.DataFrame | None = st.session_state.get("bulk_results")
    if analyzed_df is None or analyzed_df.empty:
        return

    suspicious_df = analyzed_df[analyzed_df["prediction"] == LABEL_SUSPICIOUS].copy()
    genuine_df = analyzed_df[analyzed_df["prediction"] == LABEL_GENUINE].copy()

    total_n = len(analyzed_df)
    susp_n = len(suspicious_df)
    gen_n = len(genuine_df)
    susp_rate = (susp_n / total_n * 100.0) if total_n else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Analyzed Rows", f"{total_n:,}")
    with k2:
        render_kpi_card("Potentially Suspicious", f"{susp_n:,}", f"{susp_rate:.1f}%", accent="#DC2626")
    with k3:
        render_kpi_card("Potentially Genuine", f"{gen_n:,}", f"{100.0 - susp_rate:.1f}%", accent="#16A34A")
    with k4:
        render_kpi_card("Avg Confidence", f"{analyzed_df['confidence_pct'].mean():.1f}%", accent="#2563EB")

    act1, act2, act3 = st.columns(3)
    with act1:
        st.download_button(
            "Download Full Analyzed Results (CSV)",
            data=analyzed_df.to_csv(index=False).encode("utf-8"),
            file_name="reviewshield_analyzed_results.csv",
            mime="text/csv",
            width="stretch",
        )
    with act2:
        st.download_button(
            "Download Suspicious Reviews Only (CSV)",
            data=suspicious_df.to_csv(index=False).encode("utf-8"),
            file_name="reviewshield_suspicious_only.csv",
            mime="text/csv",
            width="stretch",
        )
    with act3:
        if st.button("Save Batch to SQLite Dashboard History", width="stretch"):
            saved_n = save_bulk_records(analyzed_df, source=st.session_state.get("bulk_source_name", "bulk_csv"))
            st.success(f"Saved {saved_n} analyzed reviews to SQLite history.")

    tab_susp, tab_gen, tab_all = st.tabs([
        f"⚠️ Potentially Suspicious ({susp_n})",
        f"✅ Potentially Genuine ({gen_n})",
        f"📋 All Analyzed ({total_n})",
    ])

    display_order = [
        col
        for col in [
            "review_id",
            "product_name",
            "rating",
            "prediction",
            "confidence_pct",
            "suspicious_probability",
            "review_text",
            "explanation_summary",
        ]
        if col in analyzed_df.columns
    ]

    with tab_susp:
        st.dataframe(suspicious_df[display_order], width="stretch", height=360)
    with tab_gen:
        st.dataframe(genuine_df[display_order], width="stretch", height=360)
    with tab_all:
        st.dataframe(analyzed_df[display_order], width="stretch", height=360)


def _render_duplicate_page() -> None:
    render_hero(
        "🧬 Duplicate & Pattern Detection",
        "Uncover exact copy-paste clones, TF-IDF near-duplicate review clusters, and repeated spam phrase templates.",
    )

    input_mode = st.radio(
        "Review Source",
        options=["Use Built-in Sample Bulk CSV", "Use Current Uploaded Bulk CSV", "Paste Reviews (One Per Line)"],
        horizontal=True,
    )

    reviews: list[str] = []
    if input_mode == "Use Built-in Sample Bulk CSV":
        sample_df = pd.read_csv(BULK_SAMPLE_PATH)
        reviews = sample_df["review_text"].astype(str).tolist()
        st.caption(f"Loaded {len(reviews)} reviews from `data/bulk_test_reviews.csv`.")
    elif input_mode == "Use Current Uploaded Bulk CSV":
        bulk_df = st.session_state.get("bulk_raw_df")
        if bulk_df is None or bulk_df.empty:
            st.warning("No bulk CSV is currently loaded. Upload one in **Bulk CSV Detection** or switch source.")
            return
        col = _detect_text_column(bulk_df.columns.tolist())
        reviews = bulk_df[col].fillna("").astype(str).tolist()
        st.caption(f"Checking {len(reviews)} reviews from column `{col}`.")
    else:
        default_paste = "\n".join([
            "Great quality fast delivery five stars seller is very good recommend to all friends and family.",
            "Great quality, fast delivery, five stars! Seller is very good, recommend to all friends and family.",
            "Battery lasts about 6 hours and USB-C charging is convenient during travel.",
            "Great quality fast delivery five stars seller is very good recommend to all friends and family.",
            "DM me on whatsapp for full refund and free gift after leaving five stars review!!",
            "Contact me on whatsapp for full refund and free gift after leaving five stars review!!",
        ])
        pasted = st.text_area("Paste reviews (one review per line)", value=default_paste, height=180)
        reviews = [line.strip() for line in pasted.splitlines() if line.strip()]

    c1, c2 = st.columns(2)
    with c1:
        sim_threshold = st.slider(
            "Near-Duplicate Cosine Similarity Threshold",
            min_value=0.60,
            max_value=0.98,
            value=DEFAULT_SIMILARITY_THRESHOLD,
            step=0.02,
            help="Pairs of reviews with TF-IDF cosine similarity >= threshold are flagged as near-duplicates.",
        )
    with c2:
        min_occ = st.slider(
            "Minimum Reviews Sharing a Repeated Phrase",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
        )

    if len(reviews) < 2:
        st.info("Provide at least 2 non-empty reviews to run duplicate and phrase-pattern detection.")
        return

    dup_res = analyze_duplicates_and_patterns(
        reviews,
        similarity_threshold=sim_threshold,
        min_phrase_occurrences=min_occ,
    )
    summary = dup_res.summary

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Reviews Checked", f"{summary['total_reviews_checked']:,}")
    with k2:
        render_kpi_card(
            "Exact Duplicate Groups",
            f"{summary['exact_duplicate_groups']:,}",
            f"{summary['exact_duplicate_reviews']} total matching rows",
            accent="#DC2626",
        )
    with k3:
        render_kpi_card(
            "Near-Duplicate Pairs",
            f"{summary['near_duplicate_pairs']:,}",
            f"{summary['similarity_clusters_count']} similarity cluster(s)",
            accent="#D97706",
        )
    with k4:
        render_kpi_card(
            "Repeated Phrase Patterns",
            f"{summary['repeated_phrases_count']:,}",
            "3–4 word non-stopword n-grams",
            accent="#2563EB",
        )

    t1, t2, t3, t4 = st.tabs([
        "Exact Duplicates",
        "Near-Duplicate Pairs",
        "Similarity Clusters",
        "Repeated Suspicious Phrases",
    ])

    with t1:
        if dup_res.exact_groups.empty:
            st.success("No exact normalized duplicates found.")
        else:
            st.dataframe(dup_res.exact_groups, width="stretch")

    with t2:
        if dup_res.near_duplicate_pairs.empty:
            st.success(f"No review pairs exceeded {sim_threshold * 100:.0f}% similarity.")
        else:
            st.dataframe(dup_res.near_duplicate_pairs, width="stretch")

    with t3:
        if dup_res.template_clusters.empty:
            st.success("No multi-review similarity clusters formed at this threshold.")
        else:
            st.dataframe(dup_res.template_clusters, width="stretch")

    with t4:
        if dup_res.repeated_phrases.empty:
            st.info("No repeated 3–4 word phrases met the minimum occurrence threshold.")
        else:
            fig_phrases = px.bar(
                dup_res.repeated_phrases.head(12).sort_values("reviews_containing"),
                x="reviews_containing",
                y="phrase",
                orientation="h",
                title="Most Frequently Repeated Multi-Word Phrases Across Reviews",
                labels={"reviews_containing": "Reviews Containing Phrase", "phrase": "Repeated N-Gram"},
                color="reviews_containing",
                color_continuous_scale="Reds",
            )
            fig_phrases.update_layout(height=360, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC")
            st.plotly_chart(fig_phrases, width="stretch")
            st.dataframe(dup_res.repeated_phrases, width="stretch")


def _render_model_card_page(metadata: dict) -> None:
    render_hero(
        "🤖 ML Pipeline, Dataset & Model Card",
        "Inspect NLP preprocessing, leak-free train/test evaluation, candidate model benchmarks, and ethical boundaries.",
    )
    render_disclaimer_banner()

    audit = metadata["dataset_audit"]
    metrics = metadata["primary_metrics"]
    splits = metadata["split_sizes"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Benchmark Dataset", f"{audit['valid_rows']} rows", f"{audit['genuine_count']} genuine / {audit['suspicious_count']} suspicious")
    with k2:
        render_kpi_card("Train / Test Split", f"{splits['train_samples']} / {splits['test_samples']}", "Stratified 80/20 split")
    with k3:
        render_kpi_card("Held-Out F1 Score", f"{metrics['f1_score']:.3f}", f"5-Fold CV F1: {metrics['cv_f1_mean']:.3f}", accent="#16A34A")
    with k4:
        render_kpi_card("Held-Out ROC-AUC", f"{metrics['roc_auc']:.3f}", f"Accuracy: {metrics['accuracy'] * 100:.1f}%", accent="#2563EB")

    st.subheader("1. Preprocessing & Feature Engineering Pipeline")
    st.markdown(
        """
        1. **Text Cleaning & Normalization**: Strips HTML tags, replaces URLs/emails/phone numbers with `urltoken`, lowercases, and normalizes whitespace while retaining raw text separately for stylometric measurements.
        2. **Strict Train/Test Ordering**: Splits the dataset into 80% training and 20% held-out test sets *before* fitting vectorizers or scalers so zero test statistics leak into training.
        3. **Hybrid Feature Union**:
           - **Word N-grams (1–2) TF-IDF** (`max_features=2500`, `sublinear_tf=True`) captures vocabulary & spam phrases.
           - **Character N-grams (3–5) TF-IDF** (`analyzer="char_wb"`) captures punctuation style, capitalization patterns, and morphology.
           - **16 Standardized Stylometric Features** (`StandardScaler`) measure character/word counts, exclamation density, ALL-CAPS ratio, vocabulary diversity (TTR), promotional cue hits, balanced product-detail cues, first-person pronoun ratio, and word repetition.
        4. **Interpretable Classification**: Compares Logistic Regression, Calibrated Linear SVM, Random Forest, and Complement Naive Bayes, selecting **Logistic Regression** for calibrated probabilities and exact per-feature log-odds explainability.
        """
    )

    c1, c2 = st.columns([1.45, 1])
    with c1:
        st.subheader("2. Candidate Model Comparison (Held-Out Test)")
        comp_df = pd.DataFrame(metadata["model_comparison"])
        cols = ["model_name", "cv_f1_mean", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
        st.dataframe(comp_df[cols], width="stretch", hide_index=True)
        st.caption(f"Selected primary model: **{metadata['selected_model']}** — {metadata['selection_rationale']}")

    with c2:
        st.subheader("3. Confusion Matrix")
        fig_cm = render_confusion_matrix_figure(metrics["confusion_matrix"])
        st.pyplot(fig_cm, width="stretch")

    st.subheader("4. Documented Benchmark Dataset Preview")
    st.markdown(
        f"**Origin:** `{audit['data_origin']}` (Synthetic & curated educational dataset — **never** presented as real e-commerce ground truth).  \n"
        f"**Categories covered:** {', '.join(audit['categories'])}"
    )
    dataset_df, _ = load_and_validate_dataset(SAMPLE_DATASET_PATH)
    st.dataframe(
        dataset_df[["review_id", "product_category", "product_name", "rating", "label", "pattern_type", "review_text"]].head(25),
        width="stretch",
        height=300,
    )

    if st.button("Retrain Model Pipeline From Benchmark Dataset"):
        with st.spinner("Retraining and evaluating all candidate models..."):
            train_and_save_model()
            st.cache_resource.clear()
        st.success("Model retrained and saved successfully.")
        st.rerun()


def main() -> None:
    inject_custom_css()
    init_db()
    pipeline, metadata = get_model_bundle()

    with st.sidebar:
        st.markdown("## 🛡️ ReviewShield")
        st.caption("NLP & ML Fake Review Detection")
        page = st.radio(
            "Navigation",
            options=[
                "📊 Insights Dashboard",
                "🔍 Single Review Analysis",
                "📂 Bulk CSV Detection",
                "🧬 Duplicate & Pattern Detection",
                "🤖 ML Pipeline & Model Card",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("### Active Model")
        st.write(f"**{metadata['selected_model']}**")
        st.write(
            f"Test F1: `{metadata['primary_metrics']['f1_score']:.3f}` | "
            f"ROC-AUC: `{metadata['primary_metrics']['roc_auc']:.3f}`"
        )
        st.caption(f"Dataset: Synthetic educational benchmark ({metadata['dataset_audit']['valid_rows']} reviews).")
        st.divider()
        st.caption("Built with Python, Streamlit, Scikit-learn, Plotly & SQLite.")

    if page == "📊 Insights Dashboard":
        _render_dashboard_page(pipeline, metadata)
    elif page == "🔍 Single Review Analysis":
        _render_single_review_page(pipeline)
    elif page == "📂 Bulk CSV Detection":
        _render_bulk_page(pipeline)
    elif page == "🧬 Duplicate & Pattern Detection":
        _render_duplicate_page()
    elif page == "🤖 ML Pipeline & Model Card":
        _render_model_card_page(metadata)


if __name__ == "__main__":
    main()
