# CDAC Group Presentation Outline: Member 5 (Analytics & Feedback)

This document outlines the presentation slides for your final project demonstration.

---

### Slide 1: Title & Role Overview
* **Slide Title**: Member 5 – Analytics & Feedback Engine
* **Subtitle**: CDAC AI Mock Interview System
* **Key Content**:
  * **Role Objective**: Define a shared metric standard and consolidate scores from NLP (Member 2), Speech (Member 3), and Vision (Member 4) modules into a single, cohesive analysis framework.
  * **Core Deliverables**: Schema Design, Data Validator, Aggregation Engine, Statistical Dashboard, and Automated PDF Performance Reports.

---

### Slide 2: Schema Design & Data Validation
* **Slide Title**: Schema Standardization & Pipeline Defense
* **Key Content**:
  * **Unified Scoring Schema**:
    * Dimensions: Accuracy (NLP), Speech Confidence (Audio), Facial Dynamics (Vision).
    * Standardized format: `session_id`, `question_id`, `accuracy_score`, `speech_score`, `facial_score`.
  * **Validator Module (`validator.py`)**:
    * Performs validation checks for missing values (`NaN`).
    * Verifies data integrity and column presence to prevent system crashes during pipeline execution.

---

### Slide 3: Aggregator & Normalization Logic
* **Slide Title**: Scoring Math & Calibration Tiers
* **Key Content**:
  * **Weighted Index Formula**:
    $$\text{Overall Score} = 0.50 \times \text{Accuracy} + 0.25 \times \text{Speech} + 0.25 \times \text{Facial}$$
  * **Score Normalization**: Scales metrics from 0–100 scale to standard float range `0.0 - 1.0` to support cleaner radar plotting and visual comparisons.
  * **Feedback Calibration Rules**:
    * **$\ge$ 0.80**: Excellent performance (Green Tier).
    * **$\ge$ 0.60**: Good performance (Yellow Tier).
    * **< 0.60**: Needs practice (Red Tier).

---

### Slide 4: Statistical & Correlation Insights
* **Slide Title**: Data Analysis & Outlier Detection
* **Key Content**:
  * **Descriptive Stats**: Auto-generates mean, median, standard deviation, and variance across sessions.
  * **Pearson Correlation Analysis**: Studies inter-dependencies between verbal performance (Accuracy/Speech) and non-verbal delivery (Facial).
  * **Outlier Filtering**: Uses the Interquartile Range (IQR) method to automatically flag abnormal session inputs (e.g., hardware errors or silence).

---

### Slide 5: Automatic PDF Performance Reports
* **Slide Title**: Candidate Dashboard & PDF Generator
* **Key Content**:
  * **Dynamic ReportLab PDF Generation**: Generates `session_<id>_report.pdf` dynamically on session completion.
  * **Matplotlib Visuals**: Embeds radar charts and per-question technical accuracy bar plots.
  * **Actionable Recommendations**: Automatically highlights strengths and outlines ranked improvement feedback.
