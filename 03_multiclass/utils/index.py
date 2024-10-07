

import ee



def get_fractions(image: ee.image.Image) -> ee.image.Image:

    # default endmembers
    ENDMEMBERS = [
        [0.0119,0.0475,0.0169,0.625,0.2399,0.0675], # GV
        [0.1514,0.1597,0.1421,0.3053,0.7707,0.1975], # NPV
        [0.1799,0.2479,0.3158,0.5437,0.7707,0.6646], # Soil
        [0.4031,0.8714,0.79,0.8989,0.7002,0.6607] # Cloud
    ]


    out_bandnames = ['gv', 'npv', 'soil', 'cloud']
    
    fractions = ee.Image(image.divide(ee.Image(1000)))\
        .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])\
        .unmix(ENDMEMBERS)\
        .max(0)


    fractions = fractions.rename(out_bandnames)

    summed = fractions.expression('b("gv") + b("npv") + b("soil")')

    shade = summed.subtract(1.0).abs().rename("shade")

    fractions = fractions.addBands(shade)

    return image.addBands(fractions)


def get_ndfi(image: ee.image.Image) -> ee.image.Image:

    summed = image.expression('b("gv") + b("npv") + b("soil")')

    gvs = image.select("gv").divide(summed).rename("gvs")

    npvSoil = image.expression('b("npv") + b("soil")')

    ndfi = ee.Image.cat(gvs, npvSoil).normalizedDifference().rename('ndfi')
    
    image = image.addBands(gvs)
    image = image.addBands(ndfi)

    return image


def get_csfi(image: ee.image.Image) -> ee.image.Image:
    """Calculate CSFI and add it to image fractions

    Parameters:
        image (ee.Image): Fractions image containing the bands:
        gv, npv, soil, cloud

    Returns:
        ee.Image: Fractions image with csfi bands
    """

    csfi = image.expression(
        "(float(b('gv') - b('shade'))/(b('gv') + b('shade')))")

    csfi = csfi.rename(['csfi'])
    # csfi = csfi.multiply(100).add(100).byte().rename(['csfi'])

    image = image.addBands(csfi)

    return ee.Image(image)





