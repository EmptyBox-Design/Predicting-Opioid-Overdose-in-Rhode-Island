import Vue from 'vue'
import Vuex from 'vuex'

import { json } from 'd3-fetch'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    CBGs: {}
  },
  getters: {
    getCensusBlocks: state => {
      return state.CBGs
    }
  },
  mutations: {
    setCensusBlocks (state, CBGs) {
      state.CBGs = CBGs
    }
  },
  actions: {
    readCBGs (context) {
      json('./rhode_island_cbgs_v2.geojson', () => {}).then((response) => {
        context.commit('setCensusBlocks', response)
      })
    }
  },
  modules: {
  }
})
