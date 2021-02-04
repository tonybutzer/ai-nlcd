import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from osgeo import gdal

print(tf.__version__)
def load_ds(filepath, readonly=True):
    """
    Load a raster dataset
    """
    if not readonly:
        return gdal.Open(filepath, gdal.GA_Update)

    return gdal.Open(filepath, gdal.GA_ReadOnly)

def create(path, rows, cols, affine, datatype, proj, bands, driver='HFA'):
    """
    Create a GeoTif and return the data set to work with.
    If the file exists at the given path, this will attempt to remove it.
    """
    ds = (gdal
          .GetDriverByName(driver)
          .Create(path, cols, rows, bands, datatype, options=['COMPRESS=YES']))

    ds.SetGeoTransform(affine)
    ds.SetProjection(proj)

    return ds

def nlcd_predict(model, data, outds):
    orig_shape = data[1].shape
    print("data[1].shape=",data[1].shape)
    print("data[1].flatten().shape=",data[1].flatten().shape)
    features = np.column_stack((data[1].flatten(),data[2].flatten(),data[3].flatten(),data[4].flatten(),data[5].flatten(),data[6].flatten()))
    print(features.shape)
    preds = model.predict(features).flatten()
    print("preds.shape=",preds.shape)
    preds = np.reshape(preds, orig_shape)
    outds.GetRasterBand(1).WriteArray(preds)
    return(preds)

print("inter_op_threads=",tf.config.threading.get_inter_op_parallelism_threads())
print("intra_op_threads=",tf.config.threading.get_intra_op_parallelism_threads())
#os.sys("export OMP_NUM_THREADS=8")
#os.environ["OMP_NUM_THREADS"] = "8"
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
print("new inter_op_threads=",tf.config.threading.get_inter_op_parallelism_threads())
print("new intra_op_threads=",tf.config.threading.get_intra_op_parallelism_threads())
dnn_model = tf.keras.models.load_model('nlcd.dnn_model')

input_path = "C:\\Users\\kpostma\\OneDrive - DOI\\NLCDShrub\\nlcd_ml\\p19_r37_leafon_2019.img"
out_root = "C:\\Users\\kpostma\\OneDrive - DOI\\NLCDShrub\\nlcd_ml\\"
ds = load_ds(input_path)
if not ds:
    print("Image cannot be opened:",input_path)
geo = ds.GetGeoTransform()
proj = ds.GetProjection()

BAND_COUNT = ds.RasterCount
if BAND_COUNT <= 0:
    print("Image contains no raster bands:",input_path)

(BLK_SZ_X, BLK_SZ_Y) = ds.GetRasterBand(1).GetBlockSize()
SIZE_X = ds.RasterXSize
SIZE_Y = ds.RasterYSize
print("DS size: ("+str(SIZE_X)+","+str(SIZE_Y)+")  Block size: ("+str(BLK_SZ_X)+","+str(BLK_SZ_Y)+")")

print(ds.GetDescription())
data = {}
for idx in range(6):
    data[idx+1] = ds.GetRasterBand(idx+1).ReadAsArray()
    nodata = ds.GetRasterBand(idx+1).GetNoDataValue()
    (min,max,mean,sd) = ds.GetRasterBand(idx+1).ComputeStatistics(True)
    print("Band{}: Min={:.3f}, Max={:.3f}, Mean={:.3f}, StdDev={:.3f}, NoData={}".format(idx+1,min,max,mean,sd,nodata))

num_bands = 1
outds = create(out_root + 'nlcd_out.img', ds.RasterYSize, ds.RasterXSize, geo, gdal.GDT_Byte, proj, num_bands)

import time
start_time = time.perf_counter()
output = nlcd_predict(dnn_model, data, outds)
stop_time = time.perf_counter()
total_time = stop_time - start_time
print("Total time:",total_time,"secs")
print(output.shape)

