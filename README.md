# Predicting Opioid Overdose in Rhode Island

## Background

Opioid-related deaths have severely increased over years in the United States. In 2020, Rhode Island recorded over 350 opioid-related deaths. Since 2015, the Governor has dedicated an Overdose Prevention and Intervention Task Force to tackle the crisis. The Task Force has identified a four-pronged approach to addressing the epidemic:
- Treatment: Increase the number of people receiving medication-assisted treatment for Opioids
- Rescue: Increase the number of naloxone kits distributed across the state
- Prevention: Decrease the number of people receiving opioid & benzodiazepine subscriptions every year
- Recovery: Increase the number of peer recovery coaches and contacts

## Challenge

Rhode Island has faced challenges in effectively and equitably prioritizing communities facing the highest risk of opioid overdose events before an outbreak. The state has historically analyzed trends at the local municipality level but has been unable to create a highly spatial and temporal resolution of risk across the state. Though various interventions have been deployed in the field, for example, 25,742 naloxone kits have been distributed by the end of 2020, the increasing trend of opioid overdose deaths still continues.

## Solution

We seek to prioritize opioid overdose outbreaks in advance using predictive modeling to allocate resources amongst the state’s [Census Block Groups](https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4) (CBG). These models will predict the CBGs with the highest opioid fatality risk on a six-month rolling basis. This information will be distributed to [Rhode Island Department of Health](https://health.ri.gov/) (RIDOH) and community organizations to deploy targeted interventions to prevent overdoses. Community organizations will have the power to choose what type of intervention, such as street outreach and educational workshops, is suitable in the local neighborhood.

The modeling effort of the [New York University Center for Urban Science and Progress](https://cusp.nyu.edu/) (NYU CUSP) capstone team is part of an NIH-funded project, deemed [PROVIDENT](https://preventoverdoseri.org/research/), RIDOH, in partnership with the Task Force and researchers from Brown University and New York University (NYU).

## Evaluation Criteria

For intervention distribution fairness across localities and consistent performance comparison, all models are compared using their capture rate under the *20% lightly constrained (LC 20) scenario*: 20% of the total CBGs with the highest predicted risks, at least one CBG per town.

## Data

Our main data sources include:
- American Community Survey
- Emergency Medical Service overdoses-related runs
- Prescription Drug Monitoring Program
- Land Use
- Public access

Through literature review and expert interviews, we were able to find some of the key indicators. We further narrowed down our feature set by using Recursive Feature Elimination and Random Forest feature importances, as well as performed dimensionality reduction using Linear/Kernel Principal Component Analysis.

## Models

### Gaussian Process

The [Gaussian Process](https://scikit-learn.org/stable/modules/gaussian_process.html) (GP) method has good prospects for spatio-temporal prediction by multiplying kernels. The best GP model uses features selected by Recursive Feature Elimination, distance-weighted spatial aggregates of those features, and each Census Block Group’s centroid coordinates, in total 140 features. It captures 40.5% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for information). 

### Gradient Boost

The [XGBoost](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html) (XGB) method is highly efficient and accurate because of parallel tree boosting. The best XGB model incorporates 16 principal components extracted from an original set of 143 features, using an 8-degree poly kernel. It captures 40.1% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information). 

### Random Forest

The [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) (RF) method has good accuracy and some interoperability to help characterize the model’s *logic* via feature importance's. The best RF model uses 25 top important features from the previous two periods. It on average captures 40.2% drug overdose deaths in the periods of 2019.2 and 2020.1 (see Evaluation Criteria for detailed information). 

## Acknowledgement

We would like to thank our project sponsors, [Dr. Daniel Neill](https://cs.nyu.edu/~neill/), [Dr. Magdalena Cerda](https://med.nyu.edu/faculty/magdalena-cerda), and Bennett Allen for their advising, the Brown University team for their help with technical issues and dataset preparation, the University of California, Berkeley team for sharing their insights and CUSP instruction team for capstone guidance.

## Team

**Jiaqi Dong**

Jiaqi is a Senior Graduate Research at the NYU Furman Center of Real Estate and Urban Policy and a student at CUSP. With background in business, interactive media, and urban planning, she is interested in  developing innovative methods to help form a more equitable urban environment and moderating the conversation among various stakeholders to advocate for the underheard and marginalized communities in public engagement processes.

**Brandon Pachuca**

Brandon is an Urban Data Analyst at KPF on the KPFui team. He has a background in architecture + urban planning and has worked as a software developer creating innovative tools and workflows at the intersection of architecture and emergent technologies.
Brandon is finishing his master's at NYU Center for Urban Data Science, focusing on how technology, AI, and policy fit together to tackle our communities' challenges.

**Nicolas Liu-Sontag**

**Yicong Wang**

## Front-end Development Instructions

The front-end is built using [Vue](https://cli.vuejs.org/) and [Mapbox](https://www.mapbox.com/).

### Install Dependencies

`npm install`
### Local Server

`npm run serve`
### Deploy

[GitHub-Pages](https://emptybox-design.github.io/Predicting-Opioid-Overdose-in-Rhode-Island/)

`git subtree push --prefix dist origin gh-pages`