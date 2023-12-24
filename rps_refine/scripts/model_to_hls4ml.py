import hls4ml
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning




co = {}
_add_supported_quantized_objects(co)
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

qmodel = tf.keras.models.load_model('/Users/andyvotava/Desktop/RPS/scripts/model_refined.keras', custom_objects=co)
qmodel = strip_pruning(qmodel)

hls_config_q = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')

hls_config_q['Model']['Precision'] = 'ap_fixed<7,1>'
hls_config_q['Model']['ReuseFactor'] = 32

hls_config_q['LayerName']['activation']['Strategy'] = 'Stable'


cfg_q = hls4ml.converters.create_config(backend='Vivado')
cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg_q['HLSConfig'] = hls_config_q
cfg_q['KerasModel'] = qmodel
cfg_q['OutputDir'] = 'hls4ml_model/'
cfg_q['XilinxPart'] = 'xc7z020clg400-1'
cfg_q['Part'] = 'xc7z020clg400-1'

hls_model_q = hls4ml.converters.keras_to_hls(cfg_q)
hls_model_q.compile()


hls_model_q.build(csim=False, export=True)


print('complete')








