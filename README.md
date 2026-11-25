# HOME-CREDIT-RATING
HOME CREDIT RATING
Home Credit Default Prediction — Capstone Report
Nov 25, 2026 · 
Summary
A LightGBM model predicts loan default with an out-of-fold ROC-AUC of 0.7857, the best of five models compared on the same 5-fold cross-validation. The model uses 596 applicant-level features built from seven Home Credit tables.
•	Objective: predict whether an applicant will default (TARGET = 1) on a new loan, using the application and their credit history.
•	Data: 307,511 training applicants, of whom 24,825 (8.07%) defaulted.
•	Best model: LightGBM (ROC-AUC 0.7857, PR-AUC 0.2785), ahead of Gradient Boosting (0.7828), XGBoost (0.7827), Logistic Regression (0.7734) and Random Forest (0.7527).
•	Strongest signals: the external credit scores (EXT_SOURCE_1/2/3), employment length, and repayment history from previous loans, the credit bureau and instalments.
•	Suggested cut-off: a probability of 0.155 maximises F1. It catches 44.8% of defaulters; 26.8% of flagged applicants actually default.
Data
The project uses the Home Credit Default Risk data: one application table with the target, plus six history tables linked to applicants by SK_ID_CURR. The target is imbalanced: 8.07% defaulters.
Table	Size (MB)	Grain	Training applicants covered
application_train	166	One row per applicant (307,511)	all
bureau	170	One row per loan reported by other lenders	263,491
bureau_balance	376	One row per bureau loan per month	263,491 (via bureau)
previous_application	405	One row per earlier Home Credit application	291,057
POS_CASH_balance	393	One row per previous POS/cash loan per month	289,444
credit_card_balance	425	One row per previous credit card per month	86,905
installments_payments	723	One row per instalment payment (13.6 million)	291,643
The history tables have many rows per applicant, so each was aggregated to exactly one row per applicant before joining. They were never merged one-to-many into the application table.
Feature engineering
Each history table was turned into applicant-level features that describe volume, delinquency, amounts and timing; 465 features were added to the 122 application columns.
Source	Features	Examples
Bureau	40	Active, closed and bad-debt accounts; total, average and maximum debt; overdue days and amounts; credit dates; counts per credit type
Bureau balance	10	Months of history, delinquent and seriously delinquent months, worst status, history coverage
POS cash	9	Records, overdue records and rates, maximum and average days past due
Credit card	17	Balances, limits, utilisation (average, maximum, months above 100%), overdue months
Previous applications	363	Approval and refusal rates, amounts, instalment counts and other aggregates
Instalments	26	Days late and early, late rates, shortfalls, overpayments, longest late streak, last-12-month behaviour
Instalments: split payments. 640,905 instalments were paid in several parts. Typically a small payment comes about a month early, then the rest near the due date, and the parts add up to the amount due to within 1% in almost every case. Rows were therefore summed per instalment before measuring delay and shortfall. Without that, about 650,000 parts would have been counted as false partial payments.
Missing values carry meaning.
•	Applicants with no history in a table get a _NO_HISTORY flag. Counts are set to 0 for bureau, POS and credit-card features, following the earlier analysis.
•	Previous-application and instalment features stay missing when there is no history. LightGBM uses the missingness itself.
•	The 2,905 instalments with no payment record are counted separately, not treated as zero payments or as defaults.
•	DAYS_EMPLOYED = 365243 is a placeholder, not a real duration. It was set to missing and flagged.
Leakage controls. No feature was built from TARGET; it was used only in exploratory analysis. All history in the source tables is dated before the application.
Modelling approach
All five models were trained and scored on the same five stratified folds, so their scores are directly comparable.
•	Final table: 307,511 applicants × 596 features (580 numeric, 16 categorical), saved as home_credit_model_table.parquet.
•	Validation: 5-fold stratified cross-validation (random_state 42), keeping the 8% default rate in every fold. Each applicant is scored by the model that did not see them in training (out-of-fold predictions).
•	Metrics: ROC-AUC as the main metric; PR-AUC as a check that suits imbalanced data. Accuracy is not used, because predicting "no default" for everyone would score 92%.
Model	Role	Key settings
Logistic Regression	Baseline	Median imputation, standard scaling, one-hot categoricals, C = 0.05
Random Forest	Tree ensemble	200 trees, maximum depth 16, minimum leaf 50, half-size bootstrap samples
Gradient Boosting	Boosting comparison	scikit-learn HistGradientBoosting, learning rate 0.05, native categoricals, early stopping
XGBoost	Advanced boosting	Learning rate 0.1, depth 5, 30% of columns per tree, early stopping after 50 rounds
LightGBM	Additional comparison	Learning rate 0.05, 31 leaves, 50% of columns per tree, early stopping after 100 rounds
Two substitutions were needed on Colab's free CPU. The classic GradientBoostingClassifier is impractical at this size and cannot handle missing values, so the histogram-based version was used. XGBoost ran with faster settings after the first configuration needed over 25 minutes for a single fold.
Results
LightGBM is the best model, but the three boosting models finish within 0.003 ROC-AUC of each other. Each beats the logistic regression baseline by about 0.01, while Random Forest falls 0.02 below it. PR-AUC ranks the models in the same order, from 0.2785 for LightGBM to 0.2336 for Random Forest, against 0.081 for random guessing.
[embed: node/e185a239-f88a]
The ten most important features in LightGBM:
Rank	Feature	Source	Share of importance (%)
1	EXT_SOURCE_2	Application	12.05
2	EXT_SOURCE_3	Application	10.65
3	ORGANIZATION_TYPE	Application	5.21
4	EXT_SOURCE_1	Application	4.04
5	DAYS_EMPLOYED	Application	2.18
6	OCCUPATION_TYPE	Application	1.56
7	AMT_CREDIT	Application	1.40
8	DAYS_BIRTH	Application	1.39
9	INST_DPD_MEAN_365	Instalments	1.34
10	BUREAU_MAX_DAYS_CREDIT	Bureau	1.30
The external scores dominate. The first engineered feature, the average days late on instalments due in the last year, ranks ninth.
[embed: node/d94c51bb-6daa]
The engineered history features matter: together the six history tables account for about 47% of the model's importance. Previous applications contribute the most among them; their top features include the average number of instalments and the refusal rate.
Decision threshold
A cut-off of 0.155 maximises F1. At that point the model flags 13.5% of applicants, catches 44.8% of defaulters (11,113 of 24,825), and 26.8% of the applicants it flags actually default. The other 30,345 flagged applicants are false alarms.
Threshold	Applicants flagged (%)	Precision	Recall	F1
0.05	47.0	0.144	0.837	0.245
0.08	30.5	0.185	0.700	0.293
0.10	23.9	0.211	0.623	0.315
0.15	14.1	0.263	0.461	0.335
0.20	9.0	0.312	0.348	0.329
0.30	3.9	0.396	0.192	0.259
0.50	0.7	0.555	0.051	0.094
The right cut-off depends on what a missed default costs compared with a refused good customer. F1 weighs the two errors equally. If a default costs much more than a lost customer, a lower threshold such as 0.10 catches 62% of defaulters, at the price of flagging 24% of applicants. The 0.5 cut-off, the default in most libraries, catches only 5% of defaulters and should not be used.
Limitations and next steps
The scores are honest out-of-fold estimates, but a few choices limit how far they can be pushed.
•	Slightly optimistic boosting scores: LightGBM and XGBoost stopped early on the same fold they were scored on. The bias is small at this data size, but a separate hold-out set would remove it.
•	No tuning: every model used one reasonable configuration. Systematic tuning of LightGBM could add a little AUC.
•	High-cardinality categoricals: ORGANIZATION_TYPE (58 categories) ranks third in LightGBM importance. Native categorical splits can overfit such columns; target encoding inside the folds is an alternative to test.
•	Hardware limits: XGBoost used faster settings than the other boosters, so its score is not a like-for-like comparison.
•	Assumptions in the instalment features: payments are treated as full within ±1% of the amount due, and delay is measured to the date of the last part of each instalment.
•	No test-set scoring yet: the model has not been applied to application_test, which needs the same feature pipeline.
Suggested next steps: tune LightGBM, try a simple average of LightGBM, Gradient Boosting and XGBoost, add SHAP explanations for individual decisions, and score application_test.
Appendix: files and notebook steps
All outputs are in Google Drive, folder PROJECT DATASETS DATA MITES/HOME LOAN, and every step is a cell in home loaan rating.ipynb.
File	Contents
bureau_features.csv, bureau_balance_features.csv, pos_cash_features.csv, credit_card_features.csv, previous_application_features.csv, installments_payments_features.csv	One row per applicant for each history table
home_credit_model_table.parquet	Final modelling table: 307,511 × 597 including ID and TARGET
model_comparison.csv	The five-model comparison
oof_predictions.csv	Out-of-fold default probabilities per applicant
oof_xgboost.csv, oof_lightgbm.csv	Per-model predictions saved as each model finished
feature_importance_lightgbm.csv, feature_importance_random_forest.csv, importance_xgboost.csv	Feature importances
Notebook steps, in order:
1.	Steps 1–7: instalment EDA, feature build, validation and save.
2.	Steps 8–9: rebuild and save the bureau, bureau-balance, POS and credit-card feature tables.
3.	Step 10: build and validate the modelling table.
4.	Steps 11–15: Logistic Regression, Random Forest, Gradient Boosting, XGBoost and LightGBM.
5.	RESUME: reloads the data and saved results after a Colab disconnect, so finished models are not retrained.
6.	Step 16: comparison, feature importance, threshold analysis and saving.

