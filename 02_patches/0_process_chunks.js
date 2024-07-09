// Lista de anos para processamento
var YEARS = [
  1995, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 
  2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
  2020, 2021, 2022
];

var chunkSize = 600;

// Função para criar chunks
function createChunks(image, chunkSize) {
  var img = ee.Image(image);
  var scale = img.projection().nominalScale();
  var bounds = img.geometry().bounds();
  var xMin = bounds.coordinates().get(0).get(0);
  var yMin = bounds.coordinates().get(0).get(1);
  var xMax = bounds.coordinates().get(2).get(0);
  var yMax = bounds.coordinates().get(2).get(1);
  
  var chunks = [];
  for (var i = xMin; i < xMax; i += chunkSize) {
    for (var j = yMin; j < yMax; j += chunkSize) {
      var chunk = img.clip(ee.Geometry.Rectangle(i, j, i + chunkSize, j + chunkSize));
      chunks.push(chunk);
    }
  }
  return chunks;
}

// Função para reassemblar chunks
function reassembleChunks(chunks, originalShape, chunkSize) {
  var imgCollection = ee.ImageCollection.fromImages(chunks);
  var mosaic = imgCollection.mosaic();
  return mosaic;
}

// Função para atualizar labels
function updateLabels(prevLabels, currLabels) {
  if (prevLabels === null) {
    return currLabels;
  }
  
  // Aqui devemos definir a lógica para atualizar labels
  // Essa parte requer uma implementação detalhada com base nos requisitos específicos
  
  return currLabels;
}

var previousLabels = null;

YEARS.forEach(function(year, idx) {
  var path = 'users/yourusername/forest_' + year.toString();
  var image = ee.Image(path);
  
  var processedChunks = [];
  var chunks = createChunks(image, chunkSize);
  
  if (idx > 0) {
    var prevImage = ee.Image('users/yourusername/forest_' + (year - 1).toString());
    var prevChunks = createChunks(prevImage, chunkSize);
  } else {
    var prevChunks = [];
  }
  
  chunks.forEach(function(chunk, chunkIdx) {
    var labels = chunk.connectedComponents({connectedness: '8'});  // Conectividade 8
    var combinedArray = labels;

    if (idx > 0) {
      var prevChunk = prevChunks[chunkIdx];
      var prevLabelsChunk = previousLabels ? previousLabels[chunkIdx] : null;
      combinedArray = updateLabels(prevLabelsChunk, labels);
    }
    
    processedChunks.push(combinedArray);
  });
  
  previousLabels = processedChunks;
  var processedArray = reassembleChunks(processedChunks, image.shape, chunkSize);

  // Exportar a imagem processada
  Export.image.toDrive({
    image: processedArray,
    description: 'chunks_' + year.toString(),
    scale: 30,
    region: image.geometry(),
    crs: 'EPSG:4326'
  });
});
