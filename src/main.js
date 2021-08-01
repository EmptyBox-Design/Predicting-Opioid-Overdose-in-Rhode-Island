import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'

// BOOTSTRAP
import { BootstrapVue } from 'bootstrap-vue'
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

// MAPBOX
import '../node_modules/mapbox-gl/dist/mapbox-gl.css'
Vue.use(BootstrapVue)

Vue.config.productionTip = false

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
