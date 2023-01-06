# other packages
#import brevitas.onnx as bo
#import brevitas_examples.speech_to_text as stt

# general functions
from qonnx.core.modelwrapper import ModelWrapper
from finn.util.basic import make_build_dir
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.core.datatype import DataType
from finn.builder.build_dataflow_config import DataflowBuildConfig

# transformations
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveRandomTensorNames,
    GiveReadableTensorNames,
    GiveUniqueParameterTensors
)
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors

def step_quartznet_export(cfg: DataflowBuildConfig, model_name: str):
    """ Export QuartzNet-15x5 model from Brevitas (4-bit weights & activations)"""
    # Export model from Brevitas
    finn_onnx = cfg.output_dir + "/" + model_name + "_exported.onnx"
    quartznet_torch = stt.quant_quartznet_perchannelscaling_4b(pretrained=True, export_mode=True)
    ishape = (1, 64, 256)
    idt = DataType["FLOAT32"]
    bo.export_finn_onnx(quartznet_torch, ishape, finn_onnx)
    # Wrap in ModelWrapper
    model = ModelWrapper(finn_onnx)
    # Apply tidy-up transformations
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1

    model.save(cfg.output_dir + "/intermediate_models/" + model_name + "_exported.onnx")

#def step_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig):
#    """ Run the tidy-up step on the QuartzNet model; e.g. shape and datatype inference, constant folding, inferring node names, 3D to 4D node/tensor transformation"""
#    model = model.transform(GiveUniqueNodeNames())
#    model = model.transform(GiveRandomTensorNames())
#    model = model.transform(GiveReadableTensorNames())
#    model = model.transform(GiveUniqueParameterTensors())
#    model = model.transform(Change3DTo4DTensors())

#    return model

