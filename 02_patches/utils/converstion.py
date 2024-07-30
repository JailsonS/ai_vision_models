import xarray as xr
import rasterio
from rasterio.transform import from_origin
import numpy as np

def geotiff_to_netcdf(geotiff_path, netcdf_path):
    with rasterio.open(geotiff_path) as src:
        # ler os dados e as coordenadas
        data = src.read(1)  # Assumindo que o GeoTIFF tem apenas uma banda

        # obter as coordenadas
        transform = src.transform
        width = src.width
        height = src.height

        # gerar as coordenadas de latitude e longitude
        lon = np.arange(width) * transform[0] + transform[2]
        lat = np.arange(height) * transform[4] + transform[5]

    # criar um DataArray com xarray
    da = xr.DataArray(
        data,
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
        attrs={'crs': src.crs.to_string(), 'transform': src.transform}
    )

    # converter para um dataset (opcional, mas recomendável para NetCDF)
    ds = da.to_dataset(name='variable_name')

    # salvar o dataset como netCDF
    ds.to_netcdf(netcdf_path)




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