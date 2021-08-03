<template>
  <div id="Map">
    <div class="left-ui text-left flex-container frosted-glass p-4">
      <div class="top-panel flex-container flex-grow-1">

        <div class="title-panel title">
          <h2>Predicting Fatal Opioid Overdose</h2>
          <h4></h4>
        </div>

        <div class="body-panel text-justify">
          <p>
            We seek to prioritize opioid overdose outbreaks in advance using predictive modeling to allocate resources amongst the state’s <a href="https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4" target="blank">Census Block Groups (CBG)</a>. These models will predict the CBGs with the highest opioid fatality risk on a six-month rolling basis. This information will be distributed to <a href="https://health.ri.gov/" target="blank">Rhode Island Department of Health (RIDOH)</a> and community organizations to deploy targeted interventions to prevent overdoses.
          </p>
        </div>

        <div class="action-panel d-flex flex-column w-100 text-justify">
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
                  <p v-html="buttons[index].description"></p>
                </div>
              </div>
            </div>
          </b-form-group>
        </div>

      </div>
      <div class="footer text-center ">
        <hr>
        <div class="d-flex footer-panel">
          <div>
            <a href="https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island" target="_blank"><img src="https://img.icons8.com/material-rounded/36/000000/github.png"/></a>
          </div>
          <div class="landing-logo mb-4">
            <a href="https://cusp.nyu.edu/" target="_blank">
              <img style="height: 100%; width: auto;" :src="nyu_logo" alt="">
            </a>
          </div>
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
      center: [-71.5130, 41.6343],
      selectedModel: null,
      nyu_logo: require('../assets/nyu_cusp_logo.png'),
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
          name: 'Graph Covulution Network',
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
.left-ui {
  position: absolute;
  left: 0px;
  top: 0px;

  width: 600px;
  height: 100vh;

  z-index: 10;
}

.frosted-glass {
    // frosted glass effect
  -webkit-backdrop-filter: blur(6px);
  backdrop-filter: blur(6px);
  background-color: rgba(250, 250, 250, 0.35);

  // background-color: transparent;
}
.landing-logo {
  height: 35px;
}
.footer-panel{
  justify-content: space-evenly;
}
</style>
