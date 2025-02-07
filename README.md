# [Supervised Learning : Identifying Contributing Factors for Countries With High Cancer Rates Using Classification Algorithms With Class Imbalance Treatment](https://johnpaulinepineda.github.io/Portfolio_Project_42/)

[<img src="https://img.shields.io/badge/Python-blue?logoColor=blue&labelColor=white&style=for-the-badge" alt="Python Badge"/>](https://www.python.org/) [<img src="https://img.shields.io/badge/Jupyter-blue?logoColor=blue&labelColor=white&style=for-the-badge" alt="Jupyter Badge"/>](https://jupyter.org/)

Age-standardized cancer rates are measures used to compare cancer incidence between countries while accounting for differences in age distribution. They allow for a more accurate assessment of the relative risk of cancer across populations with diverse demographic and socio-economic characteristics - enabling a more nuanced understanding of the global burden of cancer and facilitating evidence-based public health interventions. This [case study](https://johnpaulinepineda.github.io/Portfolio_Project_42/) aims to develop an interpretable classification model which could provide robust and reliable predictions of belonging to a group of countries with high cancer rates from an optimal set of observations and predictors, while addressing class imbalance and delivering accurate predictions when applied to new unseen data. Data quality assessment and model-independent feature selection were conducted on the initial dataset to identify and remove cases or variables noted with irregularities, in adddition to the subsequent preprocessing operations most suitable for the downstream analysis. Multiple classification modelling algorithms with various hyperparameter combinations were formulated using Logistic Regression, Decision Tree, Random Forest and Support Vector Machine. Class imbalance treatment including Class Weights, Upsampling with Synthetic Minority Oversampling Technique (SMOTE) and Downsampling with Condensed Nearest Neighbors (CNN) were implemented. Ensemble Learning Using Model Stacking was additionally explored. Model performance among candidate models was compared through the F1 Score which was used as the primary performance metric (among Accuracy, Precision, Recall and Area Under the Receiver Operating Characterisng Curve (AUROC) measures); evaluated internally (using K-Fold Cross Validation) and externally (using an Independent Test Set). Post-hoc exploration of the model results to provide insights on the importance, contribution and effect of the various predictors to model prediction involved model-specific (Odds Ratios) and model-agnostic (Shapley Additive Explanations) methods.

<img src="images/CaseStudy3_Summary_0.png?raw=true"/>

<img src="images/CaseStudy3_Summary_1.png?raw=true"/>

<img src="images/CaseStudy3_Summary_2.png?raw=true"/>

<img src="images/CaseStudy3_Summary_3.png?raw=true"/>

<img src="images/CaseStudy3_Summary_4.png?raw=true"/>

<img src="images/CaseStudy3_Summary_5.png?raw=true"/>

<img src="images/CaseStudy3_Summary_6.png?raw=true"/>

