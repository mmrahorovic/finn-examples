# other packages
import brevitas.onnx as bo
import brevitas_examples.speech_to_text as stt

# general functions
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import make_build_dir, gen_finn_dt_tensor
from finn.core.datatype import DataType

# transformations
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes

def step_quartznet_export(cfg: DataflowBuildConfig, str: model_name):
    # Export model from Brevitas
    finn_onnx = cfg.output_dir + "/" + model_name "_exported.onnx"
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