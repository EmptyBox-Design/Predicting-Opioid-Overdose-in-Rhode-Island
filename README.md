# Predicting Opioid Overdose in Rhode Island

![map-gif](https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island/blob/master/src/assets/capstone_map_results.gif?raw=True)

[Map Visualization](https://emptybox-design.github.io/Predicting-Opioid-Overdose-in-Rhode-Island/)
## Abstract

The opioid epidemic is one of the largest public health crises in the United States; since 1999, over 814,000 people have died from a drug overdose in the US. Rhode Island has been hit particularly hard and regularly has some of the country's highest overdose death rates. To improve forward-looking targeting of intervention efforts they seek to utilize a prediction model of areas of the state at higher risk of an overdose outbreak. As a subset of a larger team working on this effort, we developed four models to predict overdose risk at the census block group level utilizing the following algorithms: [gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html), [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [gradient boost](https://xgboost.readthedocs.io/en/latest/), and [graph convolutional network](https://arxiv.org/pdf/1812.08434.pdf). The first three of these achieved the project’s baseline performance target and the graph convolutional network we believe shows promise. Our model results will be folded into the larger project and this information will then be supplied to RIDOH and community organizations to deploy targeted resources to higher-risk areas. If this method proves successful, it could serve as a model for states and municipalities across the country to identify and target interventions to reduce overdose risk.
## Problem Definition

Opioid-related deaths have severely increased over years in the United States: annual death involving opioids have increased by four times since 2000 (National Institute on Drug Abuse, 2020). The crisis is a public health emergency; datasets to characterize the problem are only recently beginning to show value as predictors of community risk. Our literature review found health experts using various indicators to predict fatal opioid overdose events (OOEs) or deaths such as; emergency service room visits, census tract-level demographic analyses, patient medical histories, internet search terms, and prescription drug monitoring programs. The novel process of using high-resolution spatial and temporal municipal datasets has just recently been explored to predict opioid overdose events for targeted health and policy interventions in communities.

## Problem statement

In 2020, Rhode Island recorded over 350 opioid-related deaths, and the state has regularly shown some of the highest overdose death rates per capita in the country. In 2015, the Governor created the Overdose Prevention and Intervention Task Force and set a target of reducing opioid overdose deaths by one-third within three years. The Task Force has identified a four-pronged approach to addressing the epidemic:14

Treatment: Increase the number of people receiving medication-assisted treatment for Opioids.
- Rescue: Increase the number of naloxone kits distributed across the state
- Prevention: Decrease the number of people receiving opioid & benzodiazepine subscriptions every year
- Recovery: Increase the number of peer recovery coaches and contacts

## Challenge

Extracting trends and understanding areas at higher risk of overdose is critical to effectively and efficiently executing the above actions. Thus far, Rhode Island has faced challenges in effectively and equitably targeting communities facing the highest risk of opioid overdose events before an outbreak. The state has historically analyzed trends at the local municipality level but needs a highly spatial and temporal resolution of risk across the state for prioritizing interventions in the vulnerable neighborhoods. Though various interventions have been deployed in the field, for example, 25,742 naloxone kits have been distributed by the end of 2020, the increasing trend of opioid overdose deaths continues.


## Solution

We seek to predict opioid overdose outbreaks using machine learning models to allocate resources amongst the state’s [Census Block Groups](https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4) (CBG). These models will predict the CBGs with the highest opioid fatality risk on a six-month rolling basis. This information will be distributed to [Rhode Island Department of Health](https://health.ri.gov/) (RIDOH) and community organizations to deploy targeted interventions to prevent overdoses. Community organizations will have the power to choose what type of intervention, such as street outreach and educational workshops, is suitable in the local neighborhood.

The modeling effort of the [New York University Center for Urban Science and Progress](https://cusp.nyu.edu/) (NYU CUSP) capstone team is part of an NIH-funded project, deemed [PROVIDENT](https://preventoverdoseri.org/research/), RIDOH, in partnership with the Task Force and researchers from Brown University and New York University (NYU).

As a subset of PROVIDENT, our goal is to predict opioid overdose risk in Rhode Island by CBG on a six-month rolling basis. We did a comparative analysis of ensemble, gaussian processes, and deep learning models evaluating the accuracy, interpretability, and computational effectiveness in predicting the rank of each CBG’s risk of overdoses. 

![map-image](https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island/blob/master/src/assets/Capstone_workspace-06.png?raw=True)

## Evaluation Criteria

For intervention distribution fairness across localities and consistent performance comparison, all models are compared using their capture rate under the *20% lightly constrained scenario* (LC20): 20% of the total CBGs with the highest predicted risks, at least one CBG per town.

## Data

Our data sources are:
- American Community Survey
- Emergency Medical Service overdoses-related runs
- Prescription Drug Monitoring Program
- Land Use
- Public access

Through literature review and expert interviews, we were able to narrow features to  key indicators. We further narrowed down our feature set by using recursive feature elimination and random forest feature importances, as well as performed dimensionality reduction using linear/kernel principal component analysis.

## Models

### Gaussian Process

The [Gaussian Process (GP)](https://scikit-learn.org/stable/modules/gaussian_process.html) models have good prospects for spatio-temporal prediction by multiplying kernels. The best GP model uses features selected by recursive feature elimination, distance-weighted spatial aggregates of those features, and each CBG centroid coordinates, in total 140 features. It captures 40.5% of drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for information). 

### Gradient Boost

The [Gradient Boost (GB)](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html) via the XGBoost library is highly efficient and accurate because of parallel tree boosting. The best GB model incorporates 16 principal components extracted from an original set of 143 features, using an 8-degree poly kernel. It captures 40.1% of drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information).

### Random Forest

The [Random Forest (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) models have good accuracy and some interpretability to help characterize the model’s “logic” via feature importances. The best RF model uses 25 top important features from the previous two periods. It on average captures 40.2% of drug overdose deaths in the periods of 2019.2 and 2020.1 (see Evaluation Criteria for detailed information).

### Graph Convolutional Network

The Graph Convolutional Network (GCN) are good at capturing spatio-temporal relationships when a large amount of data and steps are available. The best GCN model uses features of the previous period, plus distance-weighted spatial aggregates of those features, in total 291 features. It captures 37.4% of drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information).  

## Results and Implications

GP, GB, and RF all show promise in predicting fatal OOEs by CBG. Given the similar LC20 performance of GP, GB, and RF, we are unable to recommend a single model for use in the trial. We believe given more observations, a single model may distinguish itself and we recommend the study continue to test these models with the release of more data.

The PROVIDENT team will continue modeling as more data becomes available and will finalize model selection by late September 2021 for the randomized control trial. Information from the models will be distributed to RIDOH and community organizations to deploy targeted interventions to prevent overdoses. Community organizations will have the power to choose what type of intervention, such as street outreach or educational workshops, is suitable in local neighborhoods.

## Acknowledgement

We would like to thank our project sponsors [Dr. Daniel Neill](https://cs.nyu.edu/~neill/) (New York University), [Dr. Magdalena Cerda](https://med.nyu.edu/faculty/magdalena-cerda) (New York University Langone Health), and Bennett Allen (New York University Langone Health) for their guidance and collaboration on this project. We would also like to thank the following for their expertise and assistance: [Dr. Brandon Marshall](https://vivo.brown.edu/display/bm8) (Brown University), [Dr. Will Goedel](https://vivo.brown.edu/display/wgoedel) (Brown University), Claire Pratty (Brown University), [Maxwell Krieger](https://scholar.google.com/citations?user=qgydA6wAAAAJ&hl=en) (Brown University), [Konstantin Klemmer](https://konstantinklemmer.github.io/) (University of Warwick), Abigail Cartus (Brown University), [Rhode Island Department of Health](https://health.ri.gov/), and [NYU Center for Urban Science and Progress](https://cusp.nyu.edu/). 

## Team

**Jiaqi Dong**

Jiaqi is a Senior Graduate Research at the NYU Furman Center of Real Estate and Urban Policy and a student at CUSP. With background in business, interactive media, and urban planning, she is interested in  developing innovative methods to help form a more equitable urban environment and moderating the conversation among various stakeholders to advocate for the underheard and marginalized communities in public engagement processes.

**Brandon Pachuca**

Brandon is an Urban Data Analyst at KPF on the KPFui team. He has a background in architecture + urban planning and has worked as a software developer creating innovative tools and workflows at the intersection of architecture and emergent technologies.

Brandon is finishing his master's at NYU Center for Urban Data Science, focusing on how technology, AI, and policy fit together to tackle our communities' challenges.

**Nicolas Liu-Sontag**

Nicholas is a Sustainability Manager at New York University where he works on developing and implementing NYU’s sustainability strategies and programs. He utilizes his background in green building and energy engineering, in combination with data analysis, to identify areas for carbon and energy reduction.

Nicholas is completing his master’s NYU’s Center for Urban Science and Progress focusing on how machine learning and data analysis can solve societal problems.

**Yicong Wang**

Yicong completed his bachelor’s at UC San Diego majored in economics and math. He is finishing his master’s at NYU’s Center for Urban Science and Progress, focusing on how machine learning and participatory design can be applied in automated decision-making systems.

## Front-end Development Instructions

The front-end is built using [Vue](https://cli.vuejs.org/) and [Mapbox](https://www.mapbox.com/).