<template>
  <div id="Map">
    <div class="left-ui text-left frosted-glass">
      <div class="title">
        <h2>Predicting Opioid Overdose</h2>
        <h4>NYU Center for Urban Science Capstone</h4>
      </div>
      <div class="">
        <p>
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. Quo aliquam, cumque tenetur distinctio at porro voluptas, accusantium tempore animi odit fugit totam eligendi, earum cum delectus ut assumenda rerum quaerat.
        </p>
      </div>
      <div class="d-flex flex-column w-50">
        <b-form-group label="Select a Model">
          <div class="" v-for="(btn, index) in buttons" :key="index">
            <b-form-radio v-model="selectedModel" name="some-radios" :value="btn.value" @change="updateModel(btn.value)">{{btn.name}}</b-form-radio>
            <div class="" v-if="selectedModel === buttons[index].value">
              <p>
                {{buttons[index].description}}
              </p>
            </div>
          </div>
        </b-form-group>
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
      center: [-71.4128, 41.8240],
      selectedModel: null,
      buttons: [
        {
          value: null,
          name: 'No model',
          description: ''
        },
        {
          value: 'GP',
          name: 'Guassian Process',
          description: 'GP description'
        },
        {
          value: 'GCN',
          name: 'Graph Covulution Network',
          description: 'GCN description'
        },
        {
          value: 'RF',
          name: 'Random Forest',
          description: 'RF description'
        },
        {
          value: 'GB',
          name: 'Gradient Boost',
          description: 'GB description'
        }
      ]
    }
  },
  components: { MglMap, MglNavigationControl, MglGeolocateControl },
  created () {
    // ATTACHES MAP TO INSTANCE
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
      // console.log('- censusBlockData', censusBlockData)
      // if (map.isSourceLoaded('census-block-source-data')) {
      //   map.getSource('census-block-source-data').setData(censusBlockData)
      // }

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

.left-ui {
  position: absolute;
  left: 15px;
  top: 15px;

  width: 600px;
  height: 400px;

  z-index: 10;
}

.frosted-glass {
    // frosted glass effect
  -webkit-backdrop-filter: blur(6px);
  backdrop-filter: blur(6px);
  background-color: rgba(91, 91, 91, 0.45);

  background-color: transparent;
}
</style>
