var idx = require('users/jailson/utils:index').indexSr






var assetNICFI = 'projects/planet-nicfi/assets/basemaps/americas';
var asset = 'projects/imazon-simex/DEGRADATION/freq_logging_2022_2023';
var assetSp = 'projects/imazon-simex/DEGRADATION/amostras_logging_pt';
var assetCol = 'COPERNICUS/S2_HARMONIZED';
var assetPredictions = 'projects/ee-simex/assets/classification';







var PALETTE_NDFI = 'FFFFFF,FFFCFF,FFF9FF,FFF7FF,FFF4FF,FFF2FF,FFEFFF,FFECFF,FFEAFF,FFE7FF,'+
    'FFE5FF,FFE2FF,FFE0FF,FFDDFF,FFDAFF,FFD8FF,FFD5FF,FFD3FF,FFD0FF,FFCEFF,'+
    'FFCBFF,FFC8FF,FFC6FF,FFC3FF,FFC1FF,FFBEFF,FFBCFF,FFB9FF,FFB6FF,FFB4FF,'+
    'FFB1FF,FFAFFF,FFACFF,FFAAFF,FFA7FF,FFA4FF,FFA2FF,FF9FFF,FF9DFF,FF9AFF,'+
    'FF97FF,FF95FF,FF92FF,FF90FF,FF8DFF,FF8BFF,FF88FF,FF85FF,FF83FF,FF80FF,'+
    'FF7EFF,FF7BFF,FF79FF,FF76FF,FF73FF,FF71FF,FF6EFF,FF6CFF,FF69FF,FF67FF,'+
    'FF64FF,FF61FF,FF5FFF,FF5CFF,FF5AFF,FF57FF,FF55FF,FF52FF,FF4FFF,FF4DFF,'+
    'FF4AFF,FF48FF,FF45FF,FF42FF,FF40FF,FF3DFF,FF3BFF,FF38FF,FF36FF,FF33FF,'+
    'FF30FF,FF2EFF,FF2BFF,FF29FF,FF26FF,FF24FF,FF21FF,FF1EFF,FF1CFF,FF19FF,'+
    'FF17FF,FF14FF,FF12FF,FF0FFF,FF0CFF,FF0AFF,FF07FF,FF05FF,FF02FF,FF00FF,'+
    'FF00FF,FF0AF4,FF15E9,FF1FDF,FF2AD4,FF35C9,FF3FBF,FF4AB4,FF55AA,FF5F9F,'+
    'FF6A94,FF748A,FF7F7F,FF8A74,FF946A,FF9F5F,FFAA55,FFB44A,FFBF3F,FFC935,'+
    'FFD42A,FFDF1F,FFE915,FFF40A,FFFF00,FFFF00,FFFB00,FFF700,FFF300,FFF000,'+
    'FFEC00,FFE800,FFE400,FFE100,FFDD00,FFD900,FFD500,FFD200,FFCE00,FFCA00,'+
    'FFC600,FFC300,FFBF00,FFBB00,FFB700,FFB400,FFB000,FFAC00,FFA800,FFA500,'+
    'FFA500,F7A400,F0A300,E8A200,E1A200,D9A100,D2A000,CA9F00,C39F00,BB9E00,'+
    'B49D00,AC9C00,A59C00,9D9B00,969A00,8E9900,879900,7F9800,789700,709700,'+
    '699600,619500,5A9400,529400,4B9300,439200,349100,2D9000,258F00,1E8E00,'+
    '168E00,0F8D00,078C00,008C00,008C00,008700,008300,007F00,007A00,007600,'+
    '007200,006E00,006900,006500,006100,005C00,005800,005400,005000,004C00';

var BAND_NAMES = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'ST_B10']


var LANDSAT_NEW_NAMES = [
    'blue',
    'green',
    'red',
    'nir',
    'swir1',
    'swir2',
    'pixel_qa',
    'tir'
]

var NEW_BAND_NAMES = [
    'blue','green', 'red', 'nir','swir1', 'swir2'
]




var REFENCE = ee.FeatureCollection('projects/ee-imazon-simex-2023/assets/simex2023_total_PA').map(function(feat){
  return feat.centroid()
});


var PTS = REFENCE.reduceColumns(ee.Reducer.toList(4), ['id', 'categoria', 'date','.geo']).get('list').getInfo();








var visParams = {
  bands: ['B4', 'B3', 'B2'],
  min: 350,
  max: 1985,
  gamma: 1.4,
};
    


var App = {
  
  options: {
    coord: PTS, // it starts with the first point coord
    
    featureCollection: null,
    //coordIndex:200,
    coordIndex:290,
    
    currentImage: null,
    currentLabel: null,
    
    logoImazon: {
      uri:'gs://logo_imazon/imazon_logo.b64',
      base64: null
    },
    
    minChangeVal: -0.250,
    maxChangeVal: -0.095,
    
    lastChart: null,
    lastSample: null,
    
  },
  
  interfacaApp: {
    
    init: function(){
      
      
      
      var blob = ee.Blob(App.options.logoImazon.uri);

      blob.string().evaluate(function(str){
          str = str.replace(/\n/g, '');
          App.options.logoImazon.base64 = ui.Label({
              imageUrl: str,
              style: {
                width:'300px',
                height: '111px'
              }
          });
          
          
          App.interfacaApp.tabs.add(App.interfacaApp.buttonGoToParams);
          App.interfacaApp.tabs.add(App.interfacaApp.buttonGoToChart);
          
          App.interfacaApp.panelHead.add(App.options.logoImazon.base64);
          App.interfacaApp.panelHead.add(App.interfacaApp.labelTitle);
          App.interfacaApp.panelHead.add(App.interfacaApp.tabs);
          
          
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.inputStartEd);
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttontStartEd);
          
          
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.labelParamsChange);
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.inputMin);
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.inputMax);
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttontChange);
          

          App.interfacaApp.panelContentInfo.add(App.interfacaApp.labelSkipSession);
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttonBack);
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttonSkip);
          App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttonExport);
          //this.panelLeft.add(this.buttonFinishEdition);
          
          
          App.interfacaApp.panelContent.add(App.interfacaApp.panelHead);
          App.interfacaApp.panelContent.add(App.interfacaApp.panelContentInfo);
          
    
          
          App.interfacaApp.panelMain.add(App.interfacaApp.panelMap);
          App.interfacaApp.panelMain.add(App.interfacaApp.panelTimeLapse);
          
          ui.root.widgets().remove(ui.root.widgets().get(0));
          ui.root.insert(0, App.interfacaApp.panelContent);
          ui.root.insert(1, App.interfacaApp.panelMain);
          
          var map = App.interfacaApp.panelMap.widgets().get(0);
          
          var geometries = [ee.Geometry.Point(0,0)];
          
          var drawingTools = map.drawingTools();
          
          drawingTools.addLayer(geometries, 'add logging', 'red');
          drawingTools.addLayer(geometries, 'remove logging', 'green');
          
          drawingTools.setShape('polygon')
          
          
          App.mountTimeLapse();
      });
      
      
      
      
    
    },
    
    
    panelMain: ui.Panel({
      'layout': ui.Panel.Layout.Flow('vertical'),
      'style': {'stretch': 'both'}
    }),
    
    panelContent: ui.Panel({
      'layout': ui.Panel.Layout.flow('vertical'),
      'style': {
        //'width': '350px',
        'position': 'top-left',
        'margin': '0px 0px 0px 0px',
        'backgroundColor': '#fff',
        //'border': '1px solid darkgray',
      },
    }),
    
    panelHead: ui.Panel({
      'layout': ui.Panel.Layout.Flow('vertical'),
      //'style': {'stretch': 'both'}
    }),
    
    panelContentInfo: ui.Panel({
      'layout': ui.Panel.Layout.flow('vertical'),
      'style': {
        //'width': '350px',
        'position': 'top-left',
        'margin': '0px 0px 0px 0px',
        'backgroundColor': '#fff',
        //'border': '1px solid darkgray',
      },
    }),
    
    
    
    
    labelTitle: ui.Label('Construtor de CHIPS 1.0', {
        'position': 'top-center',
        'fontWeight': 'bold',
        // 'padding': '1px',
        'fontSize': '16px'
    }),
    
    labelParamsChange: ui.Label('Parâmetros de detecção de mudança', {
        'position': 'top-center',
        //'fontWeight': 'bold',
        // 'padding': '1px',
        'fontSize': '14px'
    }),
    
    
    labelSkipSession: ui.Label('Paginação', {
        'position': 'top-center',
        //'fontWeight': 'bold',
        // 'padding': '1px',
        'fontSize': '14px'
    }),
  
  
    
    panelTimeLapse: ui.Panel({
      'layout': ui.Panel.Layout.flow('horizontal'),
      'style': {
        'stretch': 'horizontal',
        'backgroundColor': '#ffffff',
      }
    }),
    
    panelMap: ui.Panel({
      'widgets': [ui.Map()], 
      'layout': ui.Panel.Layout.Flow('horizontal'), 
      'style':{'stretch': 'both'}
    }),
    
    




    tabs: ui.Panel({
        layout: ui.Panel.Layout.flow('horizontal')
    }),
    

    
    buttontChange: ui.Button({
        "label": "Aplicar Detecção de Mudança",
        "onClick": function (button) {
            var disabled = button.getDisabled();
            if (!disabled) {
                App.applyChange();
            }
        },
        //"disabled": true,
        "style": {
            'padding': '1px',
            'stretch': 'horizontal'
        }
    }), 
    
    
    buttonGoToParams:  ui.Button({
        "label": "Parâmetros",
        "onClick": function (button) {
            var disabled = button.getDisabled();
            if (!disabled) {
                App.goToParams();
            }
        },
        "disabled": true,
        "style": {
            'padding': '1px',
            'stretch': 'horizontal'
        }
    }),
    
    buttonGoToChart:  ui.Button({
        "label": "Gráficos",
        "onClick": function (button) {
            var disabled = button.getDisabled();
            if (!disabled) {
                App.goToChart()
            }
        },
        "disabled": false,
        "style": {
            'padding': '1px',
            'stretch': 'horizontal'
        }
    }),
    
    
    
    
    
    buttontStartEd:  ui.Button({
        "label": "Começar",
        "onClick": function (button) {
            var disabled = button.getDisabled();
            if (!disabled) {
                App.startEd()
            }
        },
        "disabled": false,
        "style": {
            'padding': '1px',
            'stretch': 'horizontal'
        }
    }),
    
    
    
    
    
    
    
    
    
    
    buttonSkip: ui.Button({
      "label": ">",
      "onClick": function (button) {
        var disabled = button.getDisabled();
        
        if (!disabled) {
          ee.Number(1).evaluate(
            function (a) {
              App.skip();
            }
          );
          
          //App.auxFunctions.loadingBox();
        }
      },
      "style": {
        'padding': '1px',
        'stretch': 'horizontal',
        'position': 'bottom-right'
      }
    }),
    
    buttonExport: ui.Button({
      "label": "Exportar",
      "onClick": function (button) {
        var disabled = button.getDisabled();
        
        if (!disabled) {
          ee.Number(1).evaluate(
            function (a) {
              App.exportSample();
            }
          );
          App.auxFunctions.loadingBox();
        }
      },
      "style": {
        'padding': '1px',
        'stretch': 'horizontal',
        'position': 'bottom-right'
      }
    }),
    
    buttonFinishEdition: ui.Button({
      "label": "Finalizar edição",
      "onClick": function (button) {
        var disabled = button.getDisabled();
        
        if (!disabled) {
          ee.Number(1).evaluate(
            function (a) {
              App.finishEdition();
            }
          );
          App.auxFunctions.loadingBox();
        }
      },
      "style": {
        'padding': '1px',
        'stretch': 'horizontal',
        'position': 'bottom-right'
      }
    }),
    
    buttonBack: ui.Button({
      "label": "<",
      "onClick": function (button) {
        var disabled = button.getDisabled();
        
        if (!disabled) {
          ee.Number(1).evaluate(
            function (a) {
              App.back();
            }
          );
          App.auxFunctions.loadingBox();
        }
      },
      "style": {
        'padding': '1px',
        'stretch': 'horizontal',
        'position': 'bottom-left'
      }
    }),

    
    
    inputStartEd: ui.Textbox({
      placeholder: 'Iniciar a edição no índice: (ex:0)',
      style: {
        margin: '4px 4px 10px 10px',
        width: '95%'
      },
      onChange:  function(txt) {
        App.setStartEdition(txt);
        //    ee.Number(1).evaluate(
        //        function (a) {
        //            App.setStartEdition(txt);
        //        }
        //    );
            
        }
    }), 
    
    inputMin: ui.Textbox({
      placeholder: 'Valor min. para logging (default: -0.250)',
      style: {
        margin: '4px 4px 10px 10px',
        width: '95%'
      },
      onChange:  function(txt) {
            ee.Number(1).evaluate(
                function (a) {
                    App.setMinVal(txt);
                }
            );
            
        }
    }),    
    
    
    inputMax: ui.Textbox({
      placeholder: 'Valor max. para logging (default: -0.095)',
      style: {
        margin: '4px 4px 10px 10px',
        width: '95%'
      },
      onChange:  function(txt) {
            ee.Number(1).evaluate(
                function (a) {
                    App.setMaxVal(txt);
                }
            );
            
        }
    }),    
    
    
    
    
    chartOptions: {
      legend: 'none',
      lineWidth: 1,
      pointSize: 5,
      vAxis: {
        gridlines: {
          count: 0
        },
        viewWindow: {
          max: 100,
          min: 0
        }
      },
      chartArea: {
        left: 30,
        top: 2,
        bottom: 100,
        right: 30,
        width: '100%',
        height: '100%'
      },
      hAxis: {
        showTextEvery: 1,
        slantedTextAngle: 90,
        slantedText: true,
        textStyle: {
          color: '#000000',
          fontSize: 12,
          fontName: 'Arial',
          bold: false,
          italic: false
        }
      },
      tooltip: {
        isHtml: true,
        textStyle: {
          fontSize: 10,
        }
      },
      crosshair: {
        trigger: 'both',
        orientation: 'vertical',
        focused: {
          color: '#0000ff'
        },
        selected: {}
      },
      annotations: {
        style: 'line'
      },
      series: {
        0: {
          type: 'line',
          color: '#000000',
          pointsVisible: true
        },
        1: {
          color: '#ff0000',
          lineWidth: 0.1,
          pointsVisible: false
        },
        2: {
          color: '#ff0000',
        },
        3: {
          color: '#ff0000',
          lineWidth: 0.1,
          pointsVisible: false
        }
      }
    },
    
    chartStyle: {
      position: 'bottom-center',
      border: '1px solid grey',
      width: '100%',
      height: '140px',
      margin: '0px',
      padding: '0px',
    },
  },
  
  
  
  mountTimeLapse: function() {
    App.auxFunctions.clear();
    App.options.featureCollection = null;
    var map = App.interfacaApp.panelMap.widgets().get(0);
    
    var aoi = ee.Geometry.Point(App.options.coord[App.options.coordIndex][3]['coordinates']);
    
    //var t1 = ee.Date(App.options.coord[0][2]).advance(5, 'days')
    var t1 = '2023-07-30'
   
    var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterBounds(aoi)
      .filterDate('2022-08-01', t1);
   
   
   
    
    var featureCollection  = ee.FeatureCollection(collection)
      .map(function(feat) { return feat.set('y', 0) });
    
    App.options.featureCollection = featureCollection;
    
    var chartLapse = ui.Chart.feature.byFeature(featureCollection, 'system:index', ['y', 'y'])
      .setChartType('ColumnChart')
      .setOptions(App.interfacaApp.chartOptions);
    
    chartLapse.style().set(App.interfacaApp.chartStyle);
    chartLapse.onClick(App.timeLapseOnClick);
    
    App.interfacaApp.panelTimeLapse.clear();
    App.interfacaApp.panelTimeLapse.add(chartLapse);
  },
  
  timeLapseOnClick: function(id){
    App.auxFunctions.clear();
    
    var aoi = ee.Geometry.Point(App.options.coord[App.options.coordIndex][3]['coordinates']);
    
    var simex = ee.FeatureCollection('projects/ee-imazon-simex-2023/assets/simex2023_total_PA')
      .map(function(feat) {return feat.set('id_raster', 1)})
      //.filter(ee.Filter.eq('id', App.options.coord[App.options.coordIndex][0]))
    
    var aoiCanvas = aoi.buffer((256 / 2) * 30, 0.01).bounds();
    
    var simexImg = simex.reduceToImage(['id_raster'], ee.Reducer.first()).unmask(0).clip(aoiCanvas)
    
    

    
    
    var image = App.idx.applyScaleFactors(ee.Image(id))
        .select(BAND_NAMES, LANDSAT_NEW_NAMES);
        
    
    var dateImage = ee.Date(image.get('system:time_start'));
        
    image = App.idx.setFractions(image);
    image = App.idx.setNdfi(image).clip(aoiCanvas);
        
    
    var nicfi = ee.ImageCollection(assetNICFI)
        .filter(ee.Filter.calendarRange(
          ee.Number(dateImage.get('year')),
          ee.Number(dateImage.get('year')),
          'year'
        ))
        .filter(ee.Filter.calendarRange(
          ee.Number(dateImage.get('month')),
          ee.Number(dateImage.get('month')),
          'month'
        ))
        //.filterDate(dateImage, dateImage.advance(1, 'days')
        .filterBounds(aoi)
        .first();
        
    print(nicfi)
        
    nicfi = ee.Image(nicfi).clip(aoiCanvas);
    
    var nicfiNdvi = nicfi.normalizedDifference(['N','R'])
    
    

    
    var imaget0 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterDate('2020-08-01', '2021-10-30')
        .filter(ee.Filter.lte('CLOUD_COVER', 25))
        .filterBounds(aoi)
        .map(App.idx.applyScaleFactors)
        .select(BAND_NAMES, LANDSAT_NEW_NAMES)
        .map(App.idx.removeCloudBQA)
        .median()
        .clip(aoiCanvas)  
    
        imaget0 = App.idx.setFractions(imaget0);
        imaget0 = App.idx.setNdfi(imaget0);
    
    
    var ndfiTemporal = imaget0.select('ndfi')
        .addBands(image.select('ndfi'))
        .addBands(image.select('ndfi'))
    
    
    var imaget1 = image.select(
      ['ndfi','soil','gv','gvs','cloud','shade','npv','red','green','blue','nir','swir1'],
      ['ndfi_t1','soil_t1','gv_t1','gvs_t1','cloud_t1','shade_t1','npv_t1','red_t1','green_t1','blue_t1','nir_t1','swir1_t1']
    )
    
    imaget0 = imaget0.select(
      ['ndfi','soil','gv','gvs','cloud','shade','npv','red','green','blue','nir','swir1'],
      ['ndfi_t0','soil_t0','gv_t0','gvs_t0','cloud_t0','shade_t0','npv_t0','red_t0','green_t0','blue_t0','nir_t0','swir1_t0']
    )
    
    var sample = imaget0.addBands(imaget1)
    
    
    
    
    

    





    var pxPercentis = sample.select(['red_t1','green_t1', 'blue_t1', 'swir1_t1', 'nir_t1']).reduceRegion({
      geometry: sample.geometry(),
      scale:30,
      maxPixels:1e13,
      reducer:ee.Reducer.percentile([3, 98])
    }).getInfo();
    

    
    var min = [
      pxPercentis['blue_t1_p3'],
      pxPercentis['green_t1_p3'],
      pxPercentis['red_t1_p3']
    ]
    
    var max = [
      pxPercentis['blue_t1_p98'],
      pxPercentis['green_t1_p98'],
      pxPercentis['red_t1_p98']
    ]
            
    var min2 = [
      pxPercentis['swir1_t1_p3'],
      pxPercentis['nir_t1_p3'],
      pxPercentis['red_t1_p3']
    ]
    
    var max2 = [
      pxPercentis['swir1_t1_p98'],
      pxPercentis['nir_t1_p98'],
      pxPercentis['red_t1_p98']
    ]
    





    
    
    // set data
    App.options.currentImage = sample
    
    
    
    var map = App.interfacaApp.panelMap.widgets().get(0);
    
    map.centerObject(aoi, 12);
 
    map.addLayer(sample, {
      bands:['red_t0','green_t0', 'blue_t0'], min:min, max:max, 
    }, 'rgb t0', false)   
    
    map.addLayer(sample, {
      bands:['ndfi_t0'], min:-1, max:1, palette:PALETTE_NDFI
    }, 'ndfi t0', false)    
    
    map.addLayer(sample, {
      bands:['ndfi_t1'], min:-0.8, max:1, palette:PALETTE_NDFI
    }, 'ndfi t1')
    
    map.addLayer(sample, {
      bands:['red_t1','green_t1', 'blue_t1'], min:min, max:max,
      //gamma:0.5
    }, 'rgb t1', false)
    
    map.addLayer(sample, {
      bands:['swir1_t1','nir_t1', 'red_t1'], min:min2, max:max2, 
    }, 'swir/nir/red t1', false)
    
    map.addLayer(ndfiTemporal, {
      min:0.1, max:0.97, 
    }, 'ndfi temp')
        
    map.addLayer(nicfi, {
      bands: ['R', 'G' , 'B'],
      max:1289.14
    }, 'planet image', false)  
    
    map.addLayer(nicfiNdvi, {
      mix:0.2, max:0.88,
      palette:PALETTE_NDFI
    }, 'ndvi planet image', false)  
    
    map.addLayer(simexImg, {min:0,max:1}, 'simex', false, 0.5);
    
    
    
    var ndfiDiff = sample.select('ndfi_t1').subtract(sample.select('ndfi_t0'))
        .rename('diff')
    
    
    
    // classified image based on ndfi

    var changeClassification2 = ndfiDiff.expression(
           '(b(0) >=-0.095 && b(0) <=0.095) ? 1 :' +
           //  No forest change
           '(b(0) >=-0.250 && b(0) <=-0.095) ? 2 :' + // Logging
           '(b(0) <=-0.250) ? 3 :' + // Deforestation
           '(b(0) >=0.095) ? 4  : 0') // Vegetation regrowth
       .updateMask(sample.select('ndfi_t0').gt(0.60)); // mask out no forest

    
    var changeClassification = ndfiDiff.gte(-0.250).and(ndfiDiff.lte(-0.095))
        changeClassification = changeClassification.where(
          sample.select('cloud_t1').gt(0.05), 0
        )
    
    
    map.addLayer(changeClassification2, {
           palette: ['000000', '1eaf0c', 'ffc239', 'ff422f','74fff9']
       }, 'change classification multi', false);
    
    map.addLayer(changeClassification, {min:0,max:1, palette:['black', 'white']}, 'change classification', false)
    
    App.interfacaApp.panelContentInfo.add(App.makeChangeHistogram(ndfiDiff))
    
    App.options.lastSample = sample;
    App.options.currentLabel = changeClassification
  },
  
  
  
  
  exportSample: function(){
      App.auxFunctions.clear();
      // Verifica se existem imagens e labels para exportar
      if (App.options.currentImage && App.options.currentLabel) {
        
        var image = App.options.currentImage;
        var label = App.options.currentLabel;
        
        var layers = App.interfacaApp.panelMap.widgets().get(0).drawingTools().layers();
        
        
        
        // check fix label
        if(layers.length() !== 0) {
          
          print(layers.length())
          
          var adjGeom = layers.get(0).getEeObject();
          var adjImg = ee.Image(1).clip(adjGeom);
          
          
          label = label.where(adjImg.eq(1), 1)
          
        }

        if(layers.length() > 1) {
          
          print(layers.length())
          
          var adjGeomRm = layers.get(1).getEeObject();
          var adjImgRm = ee.Image(1).clip(adjGeomRm);
          
          
          label = label.where(adjImgRm.eq(1), 0)
          
        }
        
    
        // Combina as bandas de imagem e o label
        var exportImage = image.addBands(label.rename('label')).double();
    
        // Define os parâmetros de exportação
        var exportParams = {
          image: exportImage,
          description: 'sample_' + App.options.coordIndex,
          bucket: 'imazon',  // Substitua pelo nome do seu bucket
          fileNamePrefix: 'mapbiomas/degradation/ai_logging_dataset/chips/sample_' + App.options.coordIndex,
          //scale: 30,
          dimensions:256,
          region: image.geometry().bounds(),
          fileFormat: 'GeoTIFF',
          formatOptions: {
            cloudOptimized: true
          }
        };
    
        // Executa a exportação
        Export.image.toCloudStorage(exportParams);
        App.auxFunctions.clearGeometry();
        App.auxFunctions.clear();
        print('Exportação iniciada...');
      } else {
        print('Imagem ou label não disponível para exportação.');
      }
  },
  
  finishEdition: function(){
      
      // Cria uma FeatureCollection que ainda não foram processados
      var processedPoints = ee.FeatureCollection(
        App.options.coord.slice(App.options.coordIndex + 1, App.options.coord.length)
        .map(function(coord) {
          return ee.Feature(ee.Geometry.Point(coord[3]['coordinates']), {index: App.options.coordIndex});
        })
      );
    
      // Define os parâmetros de exportação
      var exportParams = {
        collection: processedPoints,
        description: 'exported_points',
        assetId: assetSp,  // Substitua pelo nome do seu bucket
      };
    
      
      ee.data.deleteAsset(assetSp, function() {
        // Executa a exportação
        Export.table.toAsset(exportParams);
        print('Exportação dos pontos iniciada...');
      })
    

  },
  
  
  
  
  skip: function(){
    App.auxFunctions.clear();
    App.auxFunctions.clearGeometry();
    var currentCoordIndex = App.options.coordIndex;

    if (currentCoordIndex + 1 < App.options.coord.length) {
      App.options.coordIndex += 1;
      //App.options.coord = App.options.coord[App.options.coordIndex];
      print(App.options.coordIndex)
      App.mountTimeLapse();
    } else {
      print('No more coordinates to skip to.');
    }
  },
    
  back: function() {
    App.auxFunctions.clear();
    App.auxFunctions.clearGeometry();
    var currentCoordIndex = App.options.coordIndex;
    
    if (currentCoordIndex - 1 >= 0) {
      App.options.coordIndex -= 1;
      //App.options.coord = App.options.coord[App.options.coordIndex];
      print(App.options.coordIndex);
      App.mountTimeLapse();
    } else {
      print('No more coordinates to go back to.');
    }
  },
  
  
  goToParams: function(){
    
    App.interfacaApp.panelContentInfo.clear();
    
    App.interfacaApp.buttonGoToChart.setDisabled(false);
    App.interfacaApp.buttonGoToParams.setDisabled(true);
    
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.inputStartEd);
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttontStartEd);
    
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.labelParamsChange);
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.inputMin);
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.inputMax);
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttontChange);
    
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.labelSkipSession);
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttonBack);
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttonSkip);
    App.interfacaApp.panelContentInfo.add(App.interfacaApp.buttonExport);

  },
  
  goToChart: function() {
    App.interfacaApp.panelContentInfo.clear();
    
    App.interfacaApp.buttonGoToChart.setDisabled(true);
    App.interfacaApp.buttonGoToParams.setDisabled(false);
    
    if(App.options.lastChart !== null){
        

        var chart = ui.Chart.image.histogram({
            image: App.options.lastChart,
            region: App.options.lastChart.geometry(),
            scale: 30
        })
          .setChartType('LineChart')  // Altere para LineChart
          .setOptions({
              title: 'Histograma da Diferença do NDFI',
              hAxis: {
                title: 'Count',  // O eixo horizontal agora mostra a contagem
                titleTextStyle: {italic: false, bold: true},
              },
              vAxis: {
                title: '',  // O eixo vertical mostra os valores dos bins
                titleTextStyle: {italic: false, bold: true},
              },
              colors: ['cf513e'],  // Ajuste a cor se desejar
          });

      

      
        App.interfacaApp.panelContentInfo.add(chart)
    }
    
    
    
    
  },
  
  
  
  
  makeChangeHistogram: function(ndfiDiff){
    
    App.interfacaApp.panelContentInfo.clear();

    App.interfacaApp.buttonGoToChart.setDisabled(true);
    App.interfacaApp.buttonGoToParams.setDisabled(false);

    App.interfacaApp.buttonGoToChart.setDisabled(true);
    //App.interfacaApp.buttonGoToData.setDisabled(false);
    
    var chart = ui.Chart.image.histogram({
        image: ndfiDiff,
        region: ndfiDiff.geometry(),
        scale: 30
    })
      .setChartType('LineChart')  // Altere para LineChart
      .setOptions({
          title: 'Histograma da Diferença do NDFI',
          hAxis: {
            title: 'Count',  // O eixo horizontal agora mostra a contagem
            titleTextStyle: {italic: false, bold: true},
          },
          vAxis: {
            title: '',  // O eixo vertical mostra os valores dos bins
            titleTextStyle: {italic: false, bold: true},
          },
          colors: ['cf513e'],  // Ajuste a cor se desejar
      });
      
    App.options.lastChart = ndfiDiff;
        
    return chart;

  },
  
  
  applyChange: function(){
    
    if(
      App.options.lastChart !== null && 
      App.options.minChangeVal !== 0 &&
      App.options.maxChangeVal !== 0
    ){
      
      var map = App.interfacaApp.panelMap.widgets().get(0);
      
      
      // classified image based on ndfi
      
      var changeClassification2 = App.options.lastChart.expression(
             '(b(0) >=-0.095 && b(0) <=0.095) ? 1 :' +
             //  No forest change
             '(b(0) >=' + String(App.options.minChangeVal) + ' && b(0) <= ' + String(App.options.maxChangeVal) + ') ? 2 :' + // Logging
             '(b(0) <=' + String(App.options.minChangeVal) +') ? 3 :' + // Deforestation
             '(b(0) >=0.095) ? 4  : 0') // Vegetation regrowth
         .updateMask(App.options.lastSample.select('ndfi_t0').gt(0.60)); // mask out no forest
      
      
 
        
      var changeClassification = App.options.lastChart.gte(App.options.minChangeVal).and(App.options.lastChart.lte(App.options.maxChangeVal))
          changeClassification = changeClassification.where(
            App.options.currentImage.select('cloud_t1').gt(0.05), 0
          )
      
      map.addLayer(changeClassification2, {
             palette: ['000000', '1eaf0c', 'ffc239', 'ff422f','74fff9']
         }, 'change classification multi');
      
      map.addLayer(changeClassification, {min:0, max:1}, 'new change classification')
      
      App.options.currentLabel = changeClassification

    } else {
       print('nenhum parâmetro definido')
    }
    
    
  },
  
  
  setMinVal: function(txt){
    
    App.options.minChangeVal = parseFloat(txt)
    
  },
  
  setMaxVal: function(txt){
    App.options.maxChangeVal = parseFloat(txt)
  },
  
  setStartEdition: function(txt){
    App.options.coordIndex = parseInt(txt)
    print(App.options.coordIndex)
  },
  
  startEd: function(txt){
    
    App.auxFunctions.clear();
    App.auxFunctions.clearGeometry();
    
    App.mountTimeLapse();
    
  },
  

  idx: {
    
      removeCloudBQA: function(image){
          var qaBand = image.select(['pixel_qa']);
          //muda para 3
          var cloudMask = qaBand.bitwiseAnd(1 << 3).neq(0)
              //.focal_max(obj.dilatePixels)
              .focal_max(2)
              .rename('cloudBQAMask');
          
          return image.updateMask(cloudMask.neq(1)).copyProperties(image);
      },
    
      applyScaleFactors:function (image) {
        var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
        var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
        return image.addBands(opticalBands, null, true)
                    .addBands(thermalBands, null, true);
      },
    
      setFractions: function (image){

          var ENDMEMBERS = [
              [0.0119,0.0475,0.0169,0.625,0.2399,0.0675], // GV
              [0.1514,0.1597,0.1421,0.3053,0.7707,0.1975], // NPV
              [0.1799,0.2479,0.3158,0.5437,0.7707,0.6646], // Soil
              [0.4031,0.8714,0.79,0.8989,0.7002,0.6607] // Cloud
          ]
        
          var outBandNames = ['gv', 'npv', 'soil', 'cloud']
          
          var fractions = ee.Image(image)
              .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
              .unmix(ENDMEMBERS) 
              .max(0)
              //.multiply(100) 
              //.byte() ;
          
          fractions = fractions.rename(outBandNames);
          
          var summed = fractions.expression('b("gv") + b("npv") + b("soil")');
          
          var shade = summed 
              //.subtract(100) 
              .subtract(1.0) 
              .abs() 
              .byte() 
              .rename("shade");
        
          fractions = fractions.addBands(shade);
          
          return image.addBands(fractions);
    },

      setNdfi: function (image){

          var summed = image.expression('b("gv") + b("npv") + b("soil")')
        
          var gvs = image.select("gv").divide(summed).rename("gvs");
        
          var npvSoil = image.expression('b("npv") + b("soil")');
        
          var ndfi = ee.Image.cat(gvs, npvSoil) 
              .normalizedDifference() 
              .rename('ndfi');
        
        
        
          image = image.addBands(gvs);
          image = image.addBands(ndfi)
        
          return ee.Image(image)

      },
      
      
  },

  auxFunctions: {
    loadingBox: function(){
      var map = App.interfacaApp.panelMap.widgets().get(0);
      App.interfacaApp.loadingBox = ui.Panel();
      App.interfacaApp.loadingBox.add(ui.Label('Loading...'));
      map.add(App.interfacaApp.loadingBox);
    },
    
    clear: function(){
      var map = App.interfacaApp.panelMap.widgets().get(0);
      map.layers().reset();
      var widgets = map.widgets(); 
      widgets.remove(App.interfacaApp.loadingBox);
    },
    
    clearGeometry: function() {
      
      var drawingTools = App.interfacaApp.panelMap.widgets().get(0).drawingTools();
      var layers = drawingTools.layers();
      
      layers.forEach(function(layer){
        
          var geometriesLayers = layer.geometries();
          
          geometriesLayers.forEach(function(geom){
            
            ee.Number(0).evaluate(function(){
              geometriesLayers.remove(geom)
            });

          });
      });
    }
  },

}

App.interfacaApp.init();
