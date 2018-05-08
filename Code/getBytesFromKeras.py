from tensorflow.python.tools import freeze_graph
from keras.models import load_model

#TODO define path 
path = '../TrainedModels/emotionsClassifier.hdf5'
emotionClassifier = load_model(path, compile=False)
model = emotionClassifier					
from keras import backend as K
import tensorflow as tf
print(model.output.op.name)
saver = tf.train.Saver()
saver.save(K.get_session(), 'local_save/keras_model.ckpt')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

from keras import backend as K

directory = 'local_save'
file= 'keras_model.pb'
model = emotionClassifier
frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])

tf.train.write_graph(frozen_graph, directory, file, as_text=False)

full = directory+'/'+file
freeze_graph.freeze_graph(input_graph = full,input_binary = True,input_checkpoint = directory+'/keras_model.ckpt',output_node_names = model.output.op.name,output_graph ='your_name_graph.bytes' ,clear_devices = True, initializer_nodes = "",input_saver = "",restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0")
