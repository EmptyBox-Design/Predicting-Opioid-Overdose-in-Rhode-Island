(function(e){function t(t){for(var a,r,i=t[0],c=t[1],l=t[2],u=0,p=[];u<i.length;u++)r=i[u],Object.prototype.hasOwnProperty.call(n,r)&&n[r]&&p.push(n[r][0]),n[r]=0;for(a in c)Object.prototype.hasOwnProperty.call(c,a)&&(e[a]=c[a]);d&&d(t);while(p.length)p.shift()();return s.push.apply(s,l||[]),o()}function o(){for(var e,t=0;t<s.length;t++){for(var o=s[t],a=!0,r=1;r<o.length;r++){var c=o[r];0!==n[c]&&(a=!1)}a&&(s.splice(t--,1),e=i(i.s=o[0]))}return e}var a={},n={app:0},s=[];function r(e){return i.p+"js/"+({about:"about"}[e]||e)+"."+{about:"00c8ed8c"}[e]+".js"}function i(t){if(a[t])return a[t].exports;var o=a[t]={i:t,l:!1,exports:{}};return e[t].call(o.exports,o,o.exports,i),o.l=!0,o.exports}i.e=function(e){var t=[],o=n[e];if(0!==o)if(o)t.push(o[2]);else{var a=new Promise((function(t,a){o=n[e]=[t,a]}));t.push(o[2]=a);var s,c=document.createElement("script");c.charset="utf-8",c.timeout=120,i.nc&&c.setAttribute("nonce",i.nc),c.src=r(e);var l=new Error;s=function(t){c.onerror=c.onload=null,clearTimeout(u);var o=n[e];if(0!==o){if(o){var a=t&&("load"===t.type?"missing":t.type),s=t&&t.target&&t.target.src;l.message="Loading chunk "+e+" failed.\n("+a+": "+s+")",l.name="ChunkLoadError",l.type=a,l.request=s,o[1](l)}n[e]=void 0}};var u=setTimeout((function(){s({type:"timeout",target:c})}),12e4);c.onerror=c.onload=s,document.head.appendChild(c)}return Promise.all(t)},i.m=e,i.c=a,i.d=function(e,t,o){i.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:o})},i.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},i.t=function(e,t){if(1&t&&(e=i(e)),8&t)return e;if(4&t&&"object"===typeof e&&e&&e.__esModule)return e;var o=Object.create(null);if(i.r(o),Object.defineProperty(o,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var a in e)i.d(o,a,function(t){return e[t]}.bind(null,a));return o},i.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return i.d(t,"a",t),t},i.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},i.p="/Predicting-Opioid-Overdose-in-Rhode-Island/",i.oe=function(e){throw console.error(e),e};var c=window["webpackJsonp"]=window["webpackJsonp"]||[],l=c.push.bind(c);c.push=t,c=c.slice();for(var u=0;u<c.length;u++)t(c[u]);var d=l;s.push([0,"chunk-vendors"]),o()})({0:function(e,t,o){e.exports=o("56d7")},"26f0":function(e,t,o){e.exports=o.p+"img/nyu_cusp_logo.f7eafb4c.png"},"56d7":function(e,t,o){"use strict";o.r(t);o("e260"),o("e6cf"),o("cca6"),o("a79d");var a=o("2b0e"),n=function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("div",{attrs:{id:"app"}},[o("router-view")],1)},s=[],r=(o("5c0b"),o("2877")),i={},c=Object(r["a"])(i,n,s,!1,null,null,null),l=c.exports,u=(o("d3b7"),o("3ca3"),o("ddb0"),o("8c4f")),d=function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("div",{attrs:{id:"Map"}},[o("div",{staticClass:"left-ui text-left flex-container frosted-glass p-4"},[o("div",{staticClass:"top-panel flex-container flex-grow-1"},[e._m(0),e._m(1),o("div",{staticClass:"action-panel d-flex flex-column w-100 text-justify"},[o("b-form-group",{attrs:{label:"Select a model"}},e._l(e.buttons,(function(t,a){return o("div",{key:a},[o("b-form-radio",{attrs:{name:"some-radios",value:t.value},on:{change:function(o){return e.updateModel(t.value)}},model:{value:e.selectedModel,callback:function(t){e.selectedModel=t},expression:"selectedModel"}},[e._v(e._s(t.name))]),e.selectedModel===e.buttons[a].value?o("div",{},[e.buttons[a].description.length>1?o("div",{staticClass:"info-panel"},[o("div",{staticClass:"accuracy-box d-flex"},[o("h4",{staticClass:"mt-2"},[e._v(e._s(e.buttons[a].accuracy)+"%")]),o("div",{staticClass:"info-icon m-1"},[o("a",{attrs:{href:"https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island#evaluation-criteria",target:"_blank"}},[o("svg",{staticClass:"bi bi-info-circle",attrs:{xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",fill:"currentColor",viewBox:"0 0 16 16"}},[o("path",{attrs:{d:"M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"}}),o("path",{attrs:{d:"m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z"}})])])])]),o("p",{domProps:{innerHTML:e._s(e.buttons[a].description)}})]):e._e()]):e._e()],1)})),0)],1)]),o("div",{staticClass:"footer text-center "},[o("hr"),o("div",{staticClass:"d-flex footer-panel"},[e._m(2),o("div",{staticClass:"landing-logo mb-4"},[o("a",{attrs:{href:"https://cusp.nyu.edu/",target:"_blank"}},[o("img",{staticStyle:{height:"100%",width:"auto"},attrs:{src:e.nyu_logo,alt:""}})])])])])]),o("div",{attrs:{id:"map-container"}},[o("MglMap",{attrs:{accessToken:e.accessToken,mapStyle:e.mapStyle,container:"map-parent",zoom:e.zoom,center:e.center},on:{"update:mapStyle":function(t){e.mapStyle=t},"update:map-style":function(t){e.mapStyle=t},load:e.onMapLoaded}},[o("MglNavigationControl",{attrs:{position:"top-right"}}),o("MglGeolocateControl",{attrs:{position:"top-right"}})],1)],1)])},p=[function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("div",{staticClass:"title-panel title"},[o("h2",[e._v("Predicting Fatal Opioid Overdose")]),o("h4")])},function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("div",{staticClass:"body-panel text-justify"},[o("p",[e._v(" We seek to prioritize opioid overdose outbreaks in advance using predictive modeling to allocate resources amongst the state’s "),o("a",{attrs:{href:"https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4",target:"blank"}},[e._v("Census Block Groups (CBG)")]),e._v(". These models will predict the CBGs with the highest opioid fatality risk on a six-month rolling basis. This information will be distributed to "),o("a",{attrs:{href:"https://health.ri.gov/",target:"blank"}},[e._v("Rhode Island Department of Health (RIDOH)")]),e._v(" and community organizations to deploy targeted interventions to prevent overdoses. ")])])},function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("div",[o("a",{attrs:{href:"https://github.com/EmptyBox-Design/Predicting-Opioid-Overdose-in-Rhode-Island",target:"_blank"}},[o("img",{attrs:{src:"https://img.icons8.com/material-rounded/36/000000/github.png"}})])])}],f=o("5530"),h=(o("d81d"),o("e192")),m=o.n(h),g=o("3f3d"),b=o("2f62"),v=null,y={name:"Map",data:function(){return{accessToken:"pk.eyJ1IjoiZW1wdHlib3gtZGVzaWducyIsImEiOiJjanBtM3E3ajAwbDF0NDJsa2N0OWh4M3p0In0.ZhciPKsk9UUSjUN44kJrcw",mapStyle:"mapbox://styles/emptybox-designs/ckqzy7td011jr18na6dwm3ev0",zoom:9,center:[-71.513,41.6343],selectedModel:null,nyu_logo:o("26f0"),buttons:[{value:null,name:"No model",description:"",accuracy:""},{value:"GP",name:"Guassian Process",description:"The <a href='https://scikit-learn.org/stable/modules/gaussian_process.html' target='blank'>Gaussian Process</a> (GP) method has good prospects for spatio-temporal prediction by multiplying kernels. The best GP model uses features selected by Recursive Feature Elimination, distance-weighted spatial aggregates of those features, and each Census Block Group’s centroid coordinates, in total 140 features. It captures 40.5% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for information).",accuracy:"40.1"},{value:"GCN",name:"Graph Covulution Network",description:"",accuracy:"37.4"},{value:"RF",name:"Random Forest",description:"The <a href='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html' target='blank'>Random Forest</a> RF) method has good accuracy and some interoperability to help characterize the model’s logic via feature importance's. The best RF model uses 25 top important features from the previous two periods. It on average captures 40.2% drug overdose deaths in the periods of 2019.2 and 2020.1 (see Evaluation Criteria for detailed information).",accuracy:"40.2"},{value:"GB",name:"Gradient Boost",description:"The <a href='https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html' target='blank'>XGBoost</a> (XGB) method is highly efficient and accurate because of parallel tree boosting. The best XGB model incorporates 16 principal components extracted from an original set of 143 features, using an 8-degree poly kernel. It captures 40.1% drug overdose deaths in the period of 2020.1 (see Evaluation Criteria for detailed information).",accuracy:"40.1"}]}},components:{MglMap:g["b"],MglNavigationControl:g["c"],MglGeolocateControl:g["a"]},created:function(){this.mapbox=m.a},mounted:function(){this.subscribeToStore()},computed:Object(f["a"])({},Object(b["b"])({censusBlocks:"getCensusBlocks"})),methods:{onMapLoaded:function(e){v=e.map,this.$store.dispatch("readCBGs",{})},subscribeToStore:function(){var e=this;this.censusBlockUnsubscribe=this.$store.subscribe((function(t){switch(t.type){case"setCensusBlocks":e.createCensusBlockSource(),e.addCensusBlockData();break}}))},createCensusBlockSource:function(){v.addSource("census-block-source-data",{type:"geojson",data:{type:"FeatureCollection",features:this.censusBlocks.features}})},addCensusBlockData:function(e){v.isSourceLoaded("census-block-source-data")&&v.getSource("census-block-source-data").setData(e),v.addLayer({id:"census-block-polygons",type:"fill",source:"census-block-source-data",paint:{"fill-color":"#fdfdfd","fill-outline-color":"#333333","fill-opacity":.75}})},updateModel:function(e){var t="#fdfdfd";null!==e&&(t=["match",["get",e],0,"#fdfdfd",1,"#3ad3ad","#333333"]),v.setPaintProperty("census-block-polygons","fill-color",t)}}},k=y,_=(o("811b"),Object(r["a"])(k,d,p,!1,null,null,null)),C=_.exports;a["default"].use(u["a"]);var w=[{path:"/",name:"home",component:C},{path:"/about",name:"about",component:function(){return o.e("about").then(o.bind(null,"f820"))}}],x=new u["a"]({mode:"history",base:"/Predicting-Opioid-Overdose-in-Rhode-Island/",routes:w}),B=x,O=o("9d6a");a["default"].use(b["a"]);var M=new b["a"].Store({state:{CBGs:{}},getters:{getCensusBlocks:function(e){return e.CBGs}},mutations:{setCensusBlocks:function(e,t){e.CBGs=t}},actions:{readCBGs:function(e){Object(O["a"])("./rhode_island_cbgs_v3.geojson",(function(){})).then((function(t){e.commit("setCensusBlocks",t)}))}},modules:{}}),j=o("5f5b");o("f9e3"),o("2dd8");a["default"].use(j["a"]),a["default"].config.productionTip=!1,new a["default"]({router:B,store:M,render:function(e){return e(l)}}).$mount("#app")},"5c0b":function(e,t,o){"use strict";o("9c0c")},"811b":function(e,t,o){"use strict";o("dee9")},"9c0c":function(e,t,o){},dee9:function(e,t,o){}});
//# sourceMappingURL=app.b09af632.js.map