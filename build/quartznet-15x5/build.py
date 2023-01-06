import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
from custom_steps import (
    step_quartznet_export,
    step_tidy_up
)
import os
import shutil

# Model name
model_name = "quartznet-15x5"

# Platforms currently supported / tested on
platform_name = "U250"
vitis_platform = alveo_default_platform[board]
synth_clk_period_ns = 5.0

quartznet_build_steps = [
        "step_quartznet_export",
        #"step_tidy_up",
        #"step_streamline",
        #"step_convert_to_hls",
        #"step_create_dataflow_partition",
        #"step_target_fps_parallelization",
        #"step_apply_folding_config",
        #"step_generate_estimate_reports",
        #"step_hls_codegen",
        #"step_hls_ipgen",
        #"step_set_fifo_depths",
        #"step_create_stitched_ip",
        #"step_measure_rtlsim_performance",
        #"step_out_of_context_synthesis",
        #"step_synthesize_bitfile",
        #"step_make_pynq_driver",
        #"step_deployment_package",
]

output_dir = "output_%s_%s" % (model_name, platform_name)
os.makedirs(output_dir, exist_ok=True)

cfg = build_cfg.DataflowBuildConfig(
    steps=quartznet_build_steps,
    output_dir=output_dir,
    synth_clk_period_ns=synth_clk_period_ns,
    board=platform_name,
    shell_flow_type=build_cfg.ShellFlowType.VITIS_ALVEO,
    vitis_platform=vitis_platform,
    folding_config_file = folding_config_file,
    auto_fifo_depths=True, # Enable for first round of FIFO sizing
    # enable extra performance optimizations (physopt)
    vitis_opt_strategy=build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
    generate_outputs=[
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        #build_cfg.DataflowOutputType.OOC_SYNTHESIS,
        #build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
)

step_quartznet_export(cfg, model_name)
model_file = cfg.output_dir + "intermediate_models/%s_exported.onnx" % model_name

#model_file = cfg.output_dir + "end2end_quartznet_export_dev.onnx"
#build.build_dataflow_cfg(model_file, cfg)