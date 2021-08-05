<template>
  <div id="Map">
    <div class="left-ui text-left flex-container frosted-glass p-4">
      <div class="top-panel flex-container flex-grow-1">

        <div class="title-panel title">
          <h2>Predicting Fatal Opioid Overdose</h2>
        </div>

        <div class="body-panel text-justify font-body">
          <p>
            This interactive map presents the results of models built during the 2021  <a href="https://cusp.nyu.edu/" target="_blank">NYU Center for Urban Science and Progress</a> <a href="https://cusp.nyu.edu/2021-capstones/" target="_blank">Capstone</a> as part of a joint study between New York University and Brown University. These models predict the <a href="https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4" target="blank">Census Block Groups (CBG)</a> with the highest overdose fatality risk on a six-month rolling basis. This information will be distributed to <a href="https://health.ri.gov/" target="blank">Rhode Island Department of Health (RIDOH)</a> and community organizations to deploy targeted interventions to prevent opioid overdoses. These maps present the CBGs chosen by each model for targeting for the first half of 2020.
          </p>
          <p>
            You can learn more about the project team <a href="https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island#Team" target="blank">here</a>.
          </p>
        </div>

        <div class="navigation-panel mb-2">
          <b-button variant="info" @click="exploreMode = !exploreMode">
            <span v-if="exploreMode">About</span>
            <span v-if="!exploreMode">Explore</span>
          </b-button>
        </div>

        <div class="spacer border-bottom mb-2"></div>

        <div class="explore-panel" v-if="exploreMode">

          <div class="action-panel d-flex flex-column w-100 text-justify mb-2 border-bottom">
            <b-form-group label="Select a model">
              <div class="" v-for="(btn, index) in buttons" :key="index">
                <b-form-radio v-model="selectedModel" name="some-radios" :value="btn.value" @change="updateModel(btn.value)">{{btn.name}}</b-form-radio>
                <div class="" v-if="selectedModel === buttons[index].value">
                  <div class="info-panel" v-if="buttons[index].description.length > 1">
                    <div class="accuracy-box d-flex">
                      <h4 class="mt-2">{{buttons[index].accuracy}}%</h4>
                      <div class="info-icon m-1">
                        <a href="https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island#evaluation-criteria" target="_blank">
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-info-circle" viewBox="0 0 16 16">
                            <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                            <path d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z"/>
                          </svg>
                        </a>
                      </div>
                    </div>
                    <p class="font-body" v-html="buttons[index].description"></p>
                  </div>
                </div>
              </div>
            </b-form-group>
          </div>
        </div>

        <div class="methodology-panel font-body p-3 text-justify" v-if="!exploreMode">

          <h4>Abstract</h4>
          <p>
            The opioid epidemic is one of the largest public health crises in the United States; since 1999, over 814,000 people have died from a drug overdose in the US. Rhode Island has been hit particularly hard and regularly has some of the country's highest overdose death rates. To improve forward-looking targeting of intervention efforts they seek to utilize a prediction model of areas of the state at higher risk of an overdose outbreak. As a subset of a larger team working on this effort, we developed four models to predict overdose risk at the census block group level utilizing the following algorithms: <a href="https://scikit-learn.org/stable/modules/gaussian_process.html" target="_blank">gaussian processes</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">random forest</a>, <a href="https://xgboost.readthedocs.io/en/latest/" target="_blank">gradient boost</a>, and <a href="https://arxiv.org/pdf/1812.08434.pdf" target="_blank">graph convolutional network</a>. The first three of these achieved the project’s baseline performance target and the graph convolutional network we believe shows promise. Our model results will be folded into the larger project and this information will then be supplied to RIDOH and community organizations to deploy targeted resources to higher-risk areas. If this method proves successful, it could serve as a model for states and municipalities across the country to identify and target interventions to reduce overdose risk.
          </p>

          <h4>Problem Definition</h4>
          <p>
            Opioid-related deaths have severely increased over years in the United States: annual death involving opioids have increased by four times since 2000 (National Institute on Drug Abuse, 2020). The crisis is a public health emergency; datasets to characterize the problem are only recently beginning to show value as predictors of community risk. Our literature review found health experts using various indicators to predict fatal opioid overdose events (OOEs) or deaths such as; emergency service room visits, census tract-level demographic analyses, patient medical histories, internet search terms, and prescription drug monitoring programs. The novel process of using high-resolution spatial and temporal municipal datasets has just recently been explored to predict opioid overdose events for targeted health and policy interventions in communities.
          </p>

          <h4>Problem Statement</h4>
          <p>
            In 2020, Rhode Island recorded over 350 opioid-related deaths, and the state has regularly shown some of the highest overdose death rates per capita in the country. In 2015, the Governor created the Overdose Prevention and Intervention Task Force and set a target of reducing opioid overdose deaths by one-third within three years. The Task Force has identified a four-pronged approach to addressing the epidemic:
          </p>

          <ul>
            <li>
              <strong>Treatment</strong>: Increase the number of people receiving medication-assisted treatment for Opioids.
            </li>
            <li>
              <strong>Rescue</strong>: Increase the number of naloxone kits distributed across the state
            </li>
            <li>
              <strong>Prevention</strong>: Decrease the number of people receiving opioid & benzodiazepine subscriptions every year
            </li>
            <li>
              <strong>Recovery</strong>: Increase the number of peer recovery coaches and contacts
            </li>

          </ul>

          <h4>Challenge</h4>
          <p>
            Extracting trends and understanding areas at higher risk of overdose is critical to effectively and efficiently executing the above actions. Thus far, Rhode Island has faced challenges in effectively and equitably targeting communities facing the highest risk of opioid overdose events before an outbreak. The state has historically analyzed trends at the local municipality level but needs a highly spatial and temporal resolution of risk across the state for prioritizing interventions in the vulnerable neighborhoods. Though various interventions have been deployed in the field, for example, 25,742 naloxone kits have been distributed by the end of 2020, the increasing trend of opioid overdose deaths continues.
          </p>

          <h4>Solution</h4>
          <p>
            We seek to prioritize opioid overdose outbreaks in advance using predictive modeling to allocate resources amongst the state’s <a href="https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4" target="_blank">Census Block Groups (CBG)</a>. These models will predict the CBGs with the highest opioid fatality risk on a six-month rolling basis. This information will be distributed to <a href="https://health.ri.gov/" target="_blank">Rhode Island Department of Health (RIDOH)</a> and community organizations to deploy targeted interventions to prevent overdoses. Community organizations will have the power to choose what type of intervention, such as street outreach and educational workshops, is suitable in the local neighborhood.
          </p>
          <p>
            The modeling effort of the <a href="https://cusp.nyu.edu/" target="_blank">New York University Center for Urban Science and Progress (NYU CUSP)</a> capstone team is part of an NIH-funded project, deemed PROVIDENT, RIDOH, in partnership with the Task Force and researchers from Brown University and New York University (NYU).
          </p>
          <p>
            As a subset of PROVIDENT, our goal is to predict opioid overdose risk in Rhode Island by CBG on a six-month rolling basis. We did a comparative analysis of ensemble, gaussian processes, and deep learning models evaluating the accuracy, interpretability, and computational effectiveness in predicting the rank of each CBG’s risk of overdoses.
          </p>

          <h4>Evaluation Criteria</h4>
          <p>
            For intervention distribution fairness across localities and consistent performance comparison, all models are compared using their capture rate under the 20% lightly constrained (LC 20) scenario: 20% of the total CBGs with the highest predicted risks, at least one CBG per town.
          </p>

          <h4>Data</h4>
          <p>
            Our main data sources include:
          </p>

          <ul>
            <li>
              American Community Survey
            </li>
            <li>
              Emergency Medical Service overdoses-related runs
            </li>
            <li>
              Prescription Drug Monitoring Program
            </li>
            <li>
              Land Use
            </li>
            <li>
              Public access
            </li>
          </ul>
          <p>
            Through literature review and expert interviews, we were able to find some of the key indicators. We further narrowed down our feature set by using Recursive Feature Elimination and Random Forest feature importances, as well as performed dimensionality reduction using Linear/Kernel Principal Component Analysis.
          </p>

          <h4>Models</h4>

          <strong>Gaussian Process</strong>
          <p>
            The Gaussian Process (GP) method has good prospects for spatio-temporal prediction by multiplying kernels. The best GP model uses features selected by Recursive Feature Elimination, distance-weighted spatial aggregates of those features, and each Census Block Group’s centroid coordinates, in total 140 features. It captures 40.5% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for information).
          </p>

          <strong>Gradient Boost</strong>
          <p>
            The XGBoost (XGB) method is highly efficient and accurate because of parallel tree boosting. The best XGB model incorporates 16 principal components extracted from an original set of 143 features, using an 8-degree poly kernel. It captures 40.1% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information).
          </p>

          <strong>Random Forest</strong>
          <p>
            The Random Forest (RF) method has good accuracy and some interoperability to help characterize the model’s logic via feature importance's. The best RF model uses 25 top important features from the previous two periods. It on average captures 40.2% drug overdose deaths in the periods of 2019.2 and 2020.1 (see Evaluation Criteria for detailed information).
          </p>

          <strong>Graph Convolutional Network</strong>
          <p>
            The Graph Convolutional Network (GCN) method is good at capturing spatio-temporal relationships when a large amount of data and steps are available. The best GCN model uses features of the previous period, plus distance-weighted spatial aggregates of those features, in total 291 features. It captures 37.4% of drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information).
          </p>

          <h4>Results and Implications</h4>
          <p>
            GP, XBG, and RF all show promise in predicting OOEs per CBG. Given the similar LC20 performance of GP, XGB, and RF, we are unable to recommend a single model for use in the trial. We believe given more observations, a single model may distinguish itself and we recommend the study continue to test these models with the release of more data.
          </p>

          <p>
            The PROVIDENT team will continue the modeling effort in the next few months with a few more available periods of data to finalize the model selection by late September 2021 for the randomized control trial. This information will be distributed to RIDOH and community organizations to deploy targeted interventions to prevent overdoses. Community organizations will have the power to choose what type of intervention, such as street outreach and educational workshops, is suitable in local neighborhoods.
          </p>

          <h4>
            Acknowledgements
          </h4>
          <p>
            We would like to thank our project sponsors Dr. Daniel Neill (New York University), Dr. Magdalena Cerda (New York University Langone Health), and Bennett Allen (New York University Langone Health) for their guidance and collaboration on this project. We would also like to thank the following for their expertise and assistance: Dr. Brandon Marshall (Brown University), Dr. Will Goedel (Brown University), Claire Pratty (Brown University), Maxwell Krieger (Brown University), Konstantin Klemmer (University of Warwick), Abigail Cartus (Brown University), Rhode Island Department of Health, and NYU Center for Urban Science and Progress.
          </p>
        </div>

      </div>
      <div class="footer text-center ">
        <hr>
        <div class="d-flex flex-column footer-panel">
          <div class="d-flex justify-content-around">
            <div class="">
              <a href="https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island" target="_blank"><img src="https://img.icons8.com/material-rounded/36/000000/github.png"/></a>
            </div>
            <div class="landing-logo mb-4">
              <a href="https://cusp.nyu.edu/" target="_blank">
                <img style="height: 100%; width: auto;" :src="nyu_logo" alt="">
              </a>
            </div>
          </div>
          <div class="d-flex font-sm justify-content-center align-items-center">
            Built with
            <a target="_blank" href="https://emptybox.io"><img class="ml-1 mr-1 att-img" src="@/assets/heart_icon.png" alt /></a> and
            <a href="https://vuejs.org/" target="_blank"><img class="ml-1 mr-1 att-img" src="@/assets/logo.png" alt /></a>
          </div>
        </div>
      </div>
    </div>
    <div class="legend-panel frosted-glass p-2">
      <h5>Census Block Group Legend</h5>
      <div class="legend-container d-flex font-body">
        <div class="d-flex p-2 m-1">
          <div class="legend-circle active-color mr-2"></div>
          <div>Targetted</div>
        </div>
        <div class="d-flex p-2 m-1">
          <div class="legend-circle mr-2"></div>
          <div>Untargetted</div>
        </div>
      </div>
    </div>
    <div id="map-container">
      <MglMap
        :accessToken="accessToken"
        :mapStyle.sync="mapStyle"
        container="map-parent"
        @load="onMapLoaded"
        :zoom="zoom"
        :center="center"
      >
        <MglNavigationControl position="top-right" />
        <MglGeolocateControl position="top-right" />
      </MglMap>
    </div>
</div>
</template>

<script>

import Mapbox from 'mapbox-gl'
import { MglMap, MglNavigationControl, MglGeolocateControl } from 'vue-mapbox'

import { mapGetters } from 'vuex'

let map = null

export default {
  name: 'Map',
  data () {
    return {
      accessToken:
        'pk.eyJ1IjoiZW1wdHlib3gtZGVzaWducyIsImEiOiJjanBtM3E3ajAwbDF0NDJsa2N0OWh4M3p0In0.ZhciPKsk9UUSjUN44kJrcw',
      mapStyle: 'mapbox://styles/emptybox-designs/ckqzy7td011jr18na6dwm3ev0',
      zoom: 9,
      center: [-71.8290, 41.7074],
      selectedModel: null,
      nyu_logo: require('../assets/nyu_cusp_logo.png'),
      exploreMode: true,
      buttons: [
        {
          value: null,
          name: 'No model',
          description: '',
          accuracy: ''
        },
        {
          value: 'GP_LC20',
          name: 'Guassian Process',
          description: "The <a href='https://scikit-learn.org/stable/modules/gaussian_process.html' target='blank'>Gaussian Process</a> (GP) method has good prospects for spatio-temporal prediction by multiplying kernels. The best GP model uses features selected by Recursive Feature Elimination, distance-weighted spatial aggregates of those features, and each Census Block Group’s centroid coordinates, in total 140 features. It captures 40.5% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for information).",
          accuracy: '40.1'
        },
        {
          value: 'GCN_LC20',
          name: 'Graph Convolutional Network',
          description: 'The Graph Convolutional Network (GCN) method is good at capturing spatio-temporal relationships when a large amount of data and steps are available. The best GCN model uses features of the previous period, plus distance-weighted spatial aggregates of those features, in total 291 features. It captures 37.4% of drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information).',
          accuracy: '37.4'
        },
        {
          value: 'RF_LC20',
          name: 'Random Forest',
          description: "The <a href='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html' target='blank'>Random Forest</a> RF) method has good accuracy and some interoperability to help characterize the model’s logic via feature importance's. The best RF model uses 25 top important features from the previous two periods. It on average captures 40.2% drug overdose deaths in the periods of 2019.2 and 2020.1 (see Evaluation Criteria for detailed information).",
          accuracy: '40.2'
        },
        {
          value: 'GB_LC20',
          name: 'Gradient Boost',
          description: "The <a href='https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html' target='blank'>XGBoost</a> (XGB) method is highly efficient and accurate because of parallel tree boosting. The best XGB model incorporates 16 principal components extracted from an original set of 143 features, using an 8-degree poly kernel. It captures 40.1% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information).",
          accuracy: '40.1'
        }
      ]
    }
  },
  components: { MglMap, MglNavigationControl, MglGeolocateControl },
  created () {
    this.mapbox = Mapbox
  },
  mounted () {
    this.subscribeToStore()
  },
  computed: {
    ...mapGetters({
      censusBlocks: 'getCensusBlocks'
    })
  },
  methods: {
    onMapLoaded (event) {
      map = event.map
      this.$store.dispatch('readCBGs', {})
    },
    subscribeToStore () {
      this.censusBlockUnsubscribe = this.$store.subscribe((mutation) => {
        switch (mutation.type) {
          case 'setCensusBlocks':
            this.createCensusBlockSource()
            this.addCensusBlockData()
            break
        }
      })
    },
    createCensusBlockSource () {
      map.addSource('census-block-source-data', {
        type: 'geojson',
        data: {
          'type': 'FeatureCollection',
          'features': this.censusBlocks.features
        }
      })
    },
    addCensusBlockData (censusBlockData) {
      if (map.isSourceLoaded('census-block-source-data')) {
        map.getSource('census-block-source-data').setData(censusBlockData)
      }

      map.addLayer({
        'id': 'census-block-polygons',
        'type': 'fill',
        'source': 'census-block-source-data',
        'paint': {
          'fill-color': '#fdfdfd',
          'fill-outline-color': '#333333',
          'fill-opacity': 0.75
        }
      })
    },
    /**
     * Update model on input change
     */
    updateModel (model) {
      let paint = '#fdfdfd'

      if (model !== null) {
        paint = ['match', ['get', model], 0, '#fdfdfd', 1, '#3ad3ad', '#333333']
      }
      map.setPaintProperty(
        'census-block-polygons',
        'fill-color', paint
      )
    }
  }
}
</script>

<style lang="scss">

$black: #333;

$font-sm: 0.6rem;
$font-md: 0.9rem;
$font-lg: 1rem;
$font-xlg: 1.2rem;

$legend-cirlce-size: 15px;
$legend-circle-active-color: #3ad3ad;
$legend-circle-default: #fdfdfd;

strong {
  font-weight: 900!important;
}
.font-sm {
  font-size: $font-sm;
}
.font-body  {
  color: $black;
  font-size: $font-md;
  font-weight: 300;
}
// ---------------------------------------------------------------- MAP ----------------------------------------------------------------

#Map {
  width: 100vw;
  height: 100vh;
}
.mapboxgl-canvas {
  left: 0;
}
#map-container {
  position: absolute;
  height: 100vh;
  width: 100vw;
  left: 0;
  top: 0;

  overflow: hidden;
}

.flex-container {
  display: flex;
  flex-direction: column;
}

.att-img {
  height: 15px;
  width: auto;
}

.left-ui {
  position: absolute;
  left: 0px;
  top: 0px;

  width: 40vw;
  max-width: 50vw;
  height: 100vh;

  z-index: 10;

  overflow-y: auto;
  max-height: 100vh;
}

.methodology-panel {
  overflow-y: auto;
  max-height: 35vh;
}

.frosted-glass {
  -webkit-backdrop-filter: blur(6px);
  backdrop-filter: blur(6px);
  background-color: rgba(250, 250, 250, 0.35);
}

.landing-logo {
  height: 35px;
}

.footer-panel{
  justify-content: space-evenly;
}

.border-bottom {
  border-bottom: 1px solid $black;
}

.legend-panel {

  position: absolute;
  right: 0px;
  bottom: 0px;

  width: 300px;
  z-index: 10;

  border-radius: 4px;

  .legend-container {
    justify-content: space-evenly;
    align-items: center;
  }

  .legend-circle {
    height: $legend-cirlce-size;
    width: $legend-cirlce-size;
    background: $legend-circle-default;

    &.active-color {
      background: $legend-circle-active-color;
    }

    align-self: center;
  }
}
</style>
