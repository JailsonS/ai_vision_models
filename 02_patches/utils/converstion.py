import xarray as xr
import rasterio
from rasterio.transform import from_origin
import numpy as np

import rasterio
import numpy as np
import xarray as xr

def geotiff_to_netcdf(geotiff_path, netcdf_path):
    with rasterio.open(geotiff_path) as src:
        # Ler os dados e as coordenadas
        data = src.read(1)  # Assumindo que o GeoTIFF tem apenas uma banda

        # Obter as coordenadas
        transform = src.transform
        width = src.width
        height = src.height

        # Gerar as coordenadas de latitude e longitude
        lon = np.arange(width) * transform[0] + transform[2]
        lat = np.arange(height) * transform[4] + transform[5]

    # Criar um DataArray com xarray
    da = xr.DataArray(
        data,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        attrs={'crs': src.crs.to_string(), 'transform': src.transform}
    )

    # Converter para um dataset (opcional, mas recomendável para NetCDF)
    ds = da.to_dataset(name='variable_name')

    # Definir parâmetros de compressão
    comp = dict(zlib=True, complevel=5)

    # Salvar o dataset como netCDF com compressão usando o backend netCDF4
    ds.to_netcdf(netcdf_path, engine='netcdf4', encoding={'variable_name': comp})




def netcdf_to_geotiff(netcdf_path, geotiff_path):

    ds = xr.open_dataset(netcdf_path)

    # acessar o DataArray (assumindo que o nome da variável é 'variable_name')
    da = ds['variable_name']

    # extrair os dados e as coordenadas
    data = da.values
    lat = da['lat'].values
    lon = da['lon'].values

    # calcular o transform (transformação affine) a partir das coordenadas
    transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

    with rasterio.open(
        geotiff_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=da.attrs['crs'],
        transform=transform,
    ) as dst:
        dst.write(data, 1)