# Consolidated Analytics Report: AI Mock Interview Feedback Engine

This report documents the design, architecture, metrics, and limitations of the **Analytics & Feedback Engine (Member 5)** for the CDAC AI Mock Interview System.

---

## 1. System Architecture

The Analytics & Feedback Engine acts as the central data consolidation and intelligence layer of the application. It processes score metrics from the upstream modules (NLP, Speech, and Computer Vision), performs validation, calculates overall metrics, detects outliers, and generates candidate-facing feedback reports.

```
       +----------------------------+
       |   Upstream Score Output    |
       |  (NLP, Speech, Facial ML)  |
       +--------------+-------------+
                      |
                      v
       +--------------+-------------+
       |        validator.py        | <--- Schema/Range Validation
       +--------------+-------------+
                      | Pass
                      v
       +--------------+-------------+
       |        aggregator.py       | <--- Weighted Score Computation
       +--------------+-------------+
                      |
                      v
       +--------------+-------------+
       |  normalization_and_...py   | <--- Scale 0.0 - 1.0 & Statistics
       +--------------+-------------+
                      |
                      v
       +--------------+-------------+
       |     report_generator.py    | <--- PDF Report Compilation (ReportLab)
       +----------------------------+
```

---

## 2. Unified Scoring Schema & Validation

The unified JSON/CSV schema coordinates the score exchange across all modules to protect pipeline integrity:

* **Session ID (`session_id`)**: Unique identifier for the mock interview session.
* **Question ID (`question_id`)**: Unique identifier for the question (1 to 5).
* **Accuracy Score (`accuracy_score`)**: Technical content correctness score (0.0 to 1.0 or 0-100).
* **Speech Score (`speech_score`)**: Vocabulary, filler word control, and pacing score (0.0 to 1.0 or 0-100).
* **Facial Score (`facial_score`)**: Emotion analysis and posture tracking score (0.0 to 1.0 or 0-100).

The `validator.py` script enforces that:
1. No scores contain missing (`NaN`) values.
2. All values reside inside acceptable score bounds.
3. The schema includes all required fields before running the aggregator.

---

## 3. Scoring Weights & Feedback Rules

The final score uses a weighted formula to calculate the overall candidate performance:
$$\text{Overall Score} = 0.50 \times \text{Accuracy} + 0.25 \times \text{Speech} + 0.25 \times \text{Facial}$$

### Calibration & Feedback Thresholds
Performance categories map to the overall score using the following tiers:
* **$\ge$ 80% (0.80)**: *Excellent performance*. Candidate shows high technical mastery and delivery confidence.
* **$\ge$ 60% (0.60)**: *Good performance*. Proficient understanding, but has minor improvement points.
* **< 60% (0.60)**: *Needs more practice*. Focus on theoretical concepts, articulation pacing, or non-verbal body language.

---

## 4. Statistical & Correlation Insights (Calibration Run)

Pearson correlation analysis indicates strong positive alignment between the individual dimensions and the overall index score:
* **Accuracy vs. Overall Score**: $r \approx 0.96$
* **Speech vs. Overall Score**: $r \approx 0.79$
* **Facial vs. Overall Score**: $r \approx 0.88$

Outlier analysis uses the Interquartile Range (IQR) method:
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

This isolates anomalous sessions (e.g. complete microphone mute or blank frames) to avoid skewing average statistics.

---

## 5. System Limitations

1. **Linear Weighting Model**: The current 50-25-25 distribution is a fixed static weight and does not dynamically adjust based on the type of interview (e.g., coding-heavy vs. soft-skills-heavy).
2. **Missing Longitudinal Context**: Scores are evaluated on an isolated session basis rather than tracking candidate progression across weeks.
3. **Threshold Calibration**: Tier boundaries (60%/80%) are static rather than dynamically shifting relative to historical candidate scores.

---

## 6. Integration Roadmap

When Members 2, 3, and 4 deliver their real score output:
1. Direct their output file to `sample_data/mock_scores.csv` (or configure the paths in the `__main__` block of `aggregator.py`).
2. Run `aggregator.py` to compile the aggregated scoring matrix.
3. Run `report_generator.py` to produce candidate PDFs automatically.
