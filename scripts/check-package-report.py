#!/usr/bin/env python3
import hashlib
import json
import re
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: check-package-report.py <report>")

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(
        f"error[terlan_pytorch_report_missing]: package report does not exist: {path}"
    )

try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"error[terlan_pytorch_report_invalid]: {error}") from error

if report.get("schema") != "terlan.pytorch.package-execution.v1":
    raise SystemExit("error[terlan_pytorch_report_schema]: unexpected report schema")
if report.get("status") not in {"pass", "skip", "fail"}:
    raise SystemExit("error[terlan_pytorch_report_status]: invalid package status")
if report["status"] != "pass":
    raise SystemExit(
        f"error[terlan_pytorch_report_not_passed]: package status is {report['status']}"
    )
if not re.fullmatch(r"[0-9a-f]{40}", report.get("libtorch", {}).get("build_hash", "")):
    raise SystemExit("error[terlan_pytorch_report_build_hash]: invalid LibTorch build hash")
for name in ("source_sha256", "normalized_sha256"):
    value = report.get("binding_manifest", {}).get(name, "")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SystemExit(
            f"error[terlan_pytorch_report_manifest_hash]: invalid {name}"
        )

# Reports committed under the repository must attest to the files that the
# active package actually builds.  Keep this optional for externally supplied
# reports used by downstream consumers.
repository = path.parent.parent if path.parent.name == "reports" else None
if repository is not None:
    manifest_paths = {
        "source_sha256": repository / "libtorch-cxx.json",
        "normalized_sha256": repository / "generated/bindings/native-binding-manifest.json",
    }
    for name, manifest_path in manifest_paths.items():
        if manifest_path.is_file():
            actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if actual != report["binding_manifest"][name]:
                raise SystemExit(
                    f"error[terlan_pytorch_report_manifest_stale]: {name} does not match {manifest_path}"
                )
for gate in (
    "dlpack_packet_import",
    "dlpack_truncated_payload_rejection",
    "cpu_device_placement",
    "dtype_conversion",
    "dtype_invalid_rejection",
    "training_batch_norm_execution",
    "graph_autograd_execution",
    "adam_optimizer_execution",
    "sgd_optimizer_execution",
    "ppo_training_execution",
    "torchscript_reusable_model_execution",
    "torchscript_model_stale_handle_rejection",
    "torchscript_two_input_execution",
    "torchscript_arity_rejection",
    "torchscript_tensor_pair_execution",
    "torchscript_tensor_pair_shape_rejection",
    "torchscript_float_output_execution",
    "torchscript_float_output_type_rejection",
    "torchscript_int_output_execution",
    "torchscript_int_output_type_rejection",
    "torchscript_bool_output_execution",
    "torchscript_bool_output_type_rejection",
    "torchscript_string_output_execution",
    "torchscript_string_output_type_rejection",
    "torchscript_int_list_output_execution",
    "torchscript_int_list_output_type_rejection",
    "torchscript_float_list_output_execution",
    "torchscript_float_list_output_type_rejection",
    "torchscript_bool_list_output_execution",
    "torchscript_bool_list_output_type_rejection",
    "torchscript_string_list_output_execution",
    "torchscript_string_list_output_type_rejection",
    "torchscript_optional_tensor_output_execution",
    "torchscript_optional_tensor_output_type_rejection",
    "torchscript_tensor_list_output_execution",
    "torchscript_tensor_list_output_type_rejection",
):
    if report.get("gates", {}).get(gate) != "pass":
        raise SystemExit(
            f"error[terlan_pytorch_report_required_gate]: missing passing gate {gate}"
        )
dlpack = report.get("semantics", {}).get("dlpack_import", {})
if (
    dlpack.get("shape") != [2, 2]
    or dlpack.get("exact_mse") != 0.0
    or dlpack.get("owned_copy") is not True
    or dlpack.get("truncated_payload_rejected_before_data_access") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_dlpack]: DLPack import evidence is incomplete"
    )
placement = report.get("semantics", {}).get("cpu_device_placement", {})
if (
    placement.get("placed_device_type") != 0
    or placement.get("placed_device_index") != -1
    or placement.get("exact_mse") != 0.0
    or placement.get("synchronous") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_device_placement]: CPU placement evidence is incomplete"
    )
conversion = report.get("semantics", {}).get("dtype_conversion", {})
if (
    conversion.get("source_dtype") != 7
    or conversion.get("target_dtypes") != [4, 7, 11, 6]
    or conversion.get("float_to_int") != [3.75, 3]
    or conversion.get("int_to_float") != [3, 3.0]
    or conversion.get("float_to_bool") != [3.75, True]
    or conversion.get("shape") != [2, 2]
    or conversion.get("device_type") != 0
    or conversion.get("device_index") != -1
    or conversion.get("source_unchanged") is not True
    or conversion.get("output_owned") is not True
    or conversion.get("invalid_dtype_status") != 7064
):
    raise SystemExit(
        "error[terlan_pytorch_report_dtype_conversion]: dtype conversion evidence is incomplete"
    )
training_batch_norm = report.get("semantics", {}).get("training_batch_norm", {})
if (
    training_batch_norm.get("mode") != "training"
    or training_batch_norm.get("ownership") != "functional"
    or training_batch_norm.get("output_shape") != [2, 1, 1, 2]
    or training_batch_norm.get("updated_running_mean") != 2.0
    or training_batch_norm.get("sources_unchanged") is not True
    or training_batch_norm.get("result_tensors_independently_owned") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_training_batch_norm]: training batch normalization evidence is incomplete"
    )
graph_autograd = report.get("semantics", {}).get("graph_autograd", {})
if (
    graph_autograd.get("status") != "passed"
    or graph_autograd.get("trainable_leaf") is not True
    or graph_autograd.get("backward") is not True
    or graph_autograd.get("multiple_root_backward") is not True
    or graph_autograd.get("functional_grad") is not True
    or graph_autograd.get("multiple_root_grad") is not True
    or graph_autograd.get("inference_mode_guard") is not True
    or graph_autograd.get("autocast_guard") is not True
    or graph_autograd.get("anomaly_guard") is not True
    or graph_autograd.get("warning_guard") is not True
    or graph_autograd.get("deterministic_guard") is not True
    or graph_autograd.get("float32_matmul_precision_guard") is not True
    or graph_autograd.get("default_dtype_guard") is not True
    or graph_autograd.get("copied_gradient") is not True
    or graph_autograd.get("owned_adam_state") is not True
    or graph_autograd.get("owned_sgd_state") is not True
    or graph_autograd.get("sgd_momentum") is not True
    or graph_autograd.get("sgd_checkpoint_restore") is not True
    or graph_autograd.get("ppo_minibatch_update") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_graph_autograd]: graph-autograd evidence is incomplete"
    )
for gate, status in report.get("gates", {}).items():
    if status != "pass":
        raise SystemExit(
            f"error[terlan_pytorch_report_gate]: gate {gate} has status {status}"
        )
operations = report.get("semantics", {}).get("operations", [])
if "aten::argmax" not in operations:
    raise SystemExit("error[terlan_pytorch_report_argmax_operation]: argmax is not recorded")
for operation in ("torch::autograd::grad", "torch::autograd::backward", "c10::InferenceMode", "at::autocast::AutocastCPU", "torch::autograd::AnomalyMode", "c10::WarningUtils", "at::Context::setDeterministicAlgorithms", "at::Context::setFloat32MatmulPrecision", "c10::set_default_dtype"):
    if operation not in operations:
        raise SystemExit(
            f"error[terlan_pytorch_report_autograd_operation]: {operation} is not recorded"
        )
if "aten::mse_loss" not in operations:
    raise SystemExit("error[terlan_pytorch_report_mse_operation]: mse_loss is not recorded")
if "aten::cross_entropy_loss" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_cross_entropy_operation]: cross_entropy_loss is not recorded"
    )
if "aten::mse_loss_backward" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_mse_backward_operation]: mse_loss_backward is not recorded"
    )
if "at::convolution" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_conv2d_operation]: convolution is not recorded"
    )
if "at::adaptive_avg_pool2d" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_adaptive_avg_pool2d_operation]: adaptive pooling is not recorded"
    )
if "aten::embedding" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_embedding_operation]: embedding is not recorded"
    )
if "aten::batch_norm" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_batch_norm_operation]: batch normalization is not recorded"
    )
if "aten::layer_norm" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_layer_norm_operation]: layer normalization is not recorded"
    )
if "aten::gelu" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_gelu_operation]: GELU is not recorded"
    )
if "torch::jit::load" not in operations:
    raise SystemExit(
        "error[terlan_pytorch_report_torchscript_operation]: TorchScript loading is not recorded"
    )
torchscript = report.get("semantics", {}).get("torchscript_inference", {})
if (
    torchscript.get("format") != "TorchScript"
    or torchscript.get("input") != 3.0
    or torchscript.get("output") != 6.0
    or torchscript.get("input_unchanged") is not True
    or torchscript.get("output_owned") is not True
    or torchscript.get("missing_status") != 7041
    or torchscript.get("invalid_status") != 7042
):
    raise SystemExit(
        "error[terlan_pytorch_report_torchscript_semantics]: invalid serialized-model semantics"
    )
reusable_torchscript = report.get("semantics", {}).get("torchscript_reusable_model", {})
if (
    reusable_torchscript.get("loaded_once") is not True
    or reusable_torchscript.get("inputs") != [3.0, 4.0]
    or reusable_torchscript.get("outputs") != [6.0, 8.0]
    or reusable_torchscript.get("independently_owned_outputs") is not True
    or reusable_torchscript.get("model_disposed") is not True
    or reusable_torchscript.get("thread_safety") != "thread_confined"
    or reusable_torchscript.get("stale_model_rejected") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_reusable_torchscript_semantics]: invalid reusable-model semantics"
    )
two_input_torchscript = report.get("semantics", {}).get("torchscript_two_input_model", {})
if (
    two_input_torchscript.get("model") != "left - right * 2.0"
    or two_input_torchscript.get("ordered_inputs") != [3.0, 4.0]
    or two_input_torchscript.get("output") != -5.0
    or two_input_torchscript.get("inputs_unchanged") is not True
    or two_input_torchscript.get("output_owned") is not True
    or two_input_torchscript.get("arity_mismatch_status") != 7043
):
    raise SystemExit(
        "error[terlan_pytorch_report_two_input_torchscript_semantics]: invalid two-input model semantics"
    )
tensor_pair_torchscript = report.get("semantics", {}).get("torchscript_tensor_pair_model", {})
if (
    tensor_pair_torchscript.get("model") != "(input * 2.0, input * 3.0)"
    or tensor_pair_torchscript.get("input") != 2.0
    or tensor_pair_torchscript.get("outputs") != [4.0, 6.0]
    or tensor_pair_torchscript.get("input_unchanged") is not True
    or tensor_pair_torchscript.get("pair_owned") is not True
    or tensor_pair_torchscript.get("extracted_outputs_survive_pair_disposal") is not True
    or tensor_pair_torchscript.get("non_tuple_status") != 7046
):
    raise SystemExit(
        "error[terlan_pytorch_report_tensor_pair_torchscript_semantics]: invalid tensor-pair model semantics"
    )
float_output_torchscript = report.get("semantics", {}).get("torchscript_float_output_model", {})
if (
    float_output_torchscript.get("model") != "float(input.item()) * 2.0"
    or float_output_torchscript.get("input") != 3.0
    or float_output_torchscript.get("output") != 6.0
    or float_output_torchscript.get("input_unchanged") is not True
    or float_output_torchscript.get("copied_value") is not True
    or float_output_torchscript.get("non_float_status") != 7049
):
    raise SystemExit(
        "error[terlan_pytorch_report_float_output_torchscript_semantics]: invalid float-output model semantics"
    )
int_output_torchscript = report.get("semantics", {}).get("torchscript_int_output_model", {})
if (
    int_output_torchscript.get("model") != "int(input.item()) + 3"
    or int_output_torchscript.get("input") != 4
    or int_output_torchscript.get("output") != 7
    or int_output_torchscript.get("input_unchanged") is not True
    or int_output_torchscript.get("copied_value") is not True
    or int_output_torchscript.get("non_int_status") != 7050
):
    raise SystemExit(
        "error[terlan_pytorch_report_int_output_torchscript_semantics]: invalid int-output model semantics"
    )
bool_output_torchscript = report.get("semantics", {}).get("torchscript_bool_output_model", {})
if (
    bool_output_torchscript.get("model") != "input.item() > 0"
    or bool_output_torchscript.get("input") != 1
    or bool_output_torchscript.get("output") is not True
    or bool_output_torchscript.get("input_unchanged") is not True
    or bool_output_torchscript.get("copied_value") is not True
    or bool_output_torchscript.get("non_bool_status") != 7051
):
    raise SystemExit(
        "error[terlan_pytorch_report_bool_output_torchscript_semantics]: invalid bool-output model semantics"
    )
string_output_torchscript = report.get("semantics", {}).get("torchscript_string_output_model", {})
if (
    string_output_torchscript.get("model") != "input.item() > 0 ? positive : non_positive"
    or string_output_torchscript.get("input") != 1
    or string_output_torchscript.get("output") != "positive"
    or string_output_torchscript.get("input_unchanged") is not True
    or string_output_torchscript.get("copied_value") is not True
    or string_output_torchscript.get("non_string_status") != 7052
):
    raise SystemExit(
        "error[terlan_pytorch_report_string_output_torchscript_semantics]: invalid string-output model semantics"
    )
int_list_output_torchscript = report.get("semantics", {}).get("torchscript_int_list_output_model", {})
if (
    int_list_output_torchscript.get("model") != "[value, value + 1, value + 2]"
    or int_list_output_torchscript.get("input") != 4
    or int_list_output_torchscript.get("output") != [4, 5, 6]
    or int_list_output_torchscript.get("input_unchanged") is not True
    or int_list_output_torchscript.get("copied_value") is not True
    or int_list_output_torchscript.get("non_int_list_status") != 7053
):
    raise SystemExit(
        "error[terlan_pytorch_report_int_list_output_torchscript_semantics]: invalid integer-list-output model semantics"
    )
float_list_output_torchscript = report.get("semantics", {}).get("torchscript_float_list_output_model", {})
if (
    float_list_output_torchscript.get("model") != "[value, value + 0.5, value + 1.0]"
    or float_list_output_torchscript.get("input") != 1.5
    or float_list_output_torchscript.get("output") != [1.5, 2.0, 2.5]
    or float_list_output_torchscript.get("input_unchanged") is not True
    or float_list_output_torchscript.get("copied_value") is not True
    or float_list_output_torchscript.get("non_float_list_status") != 7054
):
    raise SystemExit(
        "error[terlan_pytorch_report_float_list_output_torchscript_semantics]: invalid floating-point-list-output model semantics"
    )
bool_list_output_torchscript = report.get("semantics", {}).get("torchscript_bool_list_output_model", {})
if (
    bool_list_output_torchscript.get("model") != "[value > 0, value == 0, value < 0]"
    or bool_list_output_torchscript.get("input") != 2
    or bool_list_output_torchscript.get("output") != [True, False, False]
    or bool_list_output_torchscript.get("input_unchanged") is not True
    or bool_list_output_torchscript.get("copied_value") is not True
    or bool_list_output_torchscript.get("portable_encoding") != "uint8-0-or-1"
    or bool_list_output_torchscript.get("non_bool_list_status") != 7055
):
    raise SystemExit(
        "error[terlan_pytorch_report_bool_list_output_torchscript_semantics]: invalid boolean-list-output model semantics"
    )
string_list_output_torchscript = report.get("semantics", {}).get("torchscript_string_list_output_model", {})
if (
    string_list_output_torchscript.get("model") != "input > 0 ? [positive, empty, ready] : [non_positive, empty, waiting]"
    or string_list_output_torchscript.get("input") != 1
    or string_list_output_torchscript.get("output") != ["positive", "", "ready"]
    or string_list_output_torchscript.get("input_unchanged") is not True
    or string_list_output_torchscript.get("copied_value") is not True
    or string_list_output_torchscript.get("encoding") != "length-delimited-utf8"
    or string_list_output_torchscript.get("non_string_list_status") != 7056
):
    raise SystemExit(
        "error[terlan_pytorch_report_string_list_output_torchscript_semantics]: invalid string-list-output model semantics"
    )
optional_tensor_torchscript = report.get("semantics", {}).get(
    "torchscript_optional_tensor_output_model", {}
)
if (
    optional_tensor_torchscript.get("present_input") != 3.0
    or optional_tensor_torchscript.get("present_output") != 6.0
    or optional_tensor_torchscript.get("absent_input") != 0.0
    or optional_tensor_torchscript.get("absent_output") != "None"
    or optional_tensor_torchscript.get("inputs_unchanged") is not True
    or optional_tensor_torchscript.get("present_tensor_owned") is not True
    or optional_tensor_torchscript.get("non_optional_status") != 7057
):
    raise SystemExit(
        "error[terlan_pytorch_report_optional_tensor_output_torchscript_semantics]: invalid optional-tensor-output model semantics"
    )
tensor_list_torchscript = report.get("semantics", {}).get(
    "torchscript_tensor_list_output_model", {}
)
if (
    tensor_list_torchscript.get("model") != "[input * 2.0, input]"
    or tensor_list_torchscript.get("input") != 3.0
    or tensor_list_torchscript.get("output") != [6.0, 3.0]
    or tensor_list_torchscript.get("input_unchanged") is not True
    or tensor_list_torchscript.get("elements_independently_owned") is not True
    or tensor_list_torchscript.get("temporary_handle_array_destroyed") is not True
    or tensor_list_torchscript.get("non_tensor_list_status") != 7058
):
    raise SystemExit(
        "error[terlan_pytorch_report_tensor_list_output_torchscript_semantics]: invalid tensor-list-output model semantics"
    )
classifier = report.get("semantics", {}).get("classifier_forward", {})
if (
    classifier.get("prediction_shape") != [2]
    or classifier.get("prediction_dtype") != 4
    or classifier.get("prediction_maximum") != 0
):
    raise SystemExit(
        "error[terlan_pytorch_report_classifier_prediction]: invalid classifier prediction semantics"
    )
if classifier.get("mse_loss") != 0.25 or classifier.get("loss_reduction") != 1:
    raise SystemExit(
        "error[terlan_pytorch_report_classifier_loss]: invalid classifier loss semantics"
    )
if (
    classifier.get("class_target_shape") != [2]
    or classifier.get("label_smoothing") != 0.1
    or classifier.get("cross_entropy_loss_min") != 0.69
    or classifier.get("cross_entropy_loss_max") != 0.70
):
    raise SystemExit(
        "error[terlan_pytorch_report_classifier_cross_entropy]: invalid classification loss semantics"
    )
gradient_step = report.get("semantics", {}).get("gradient_step", {})
if (
    gradient_step.get("gradient") != 0.5
    or gradient_step.get("loss_before") != 1.0
    or gradient_step.get("loss_decreased") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_gradient_step]: invalid explicit gradient-step semantics"
    )
conv2d = report.get("semantics", {}).get("conv2d", {})
if (
    conv2d.get("layout") != "NCHW"
    or conv2d.get("output_shape") != [1, 1, 2, 2]
    or conv2d.get("output_value") != 4.0
    or conv2d.get("exact_mse") != 0.0
    or conv2d.get("sources_unchanged") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_conv2d]: invalid convolution semantics"
    )
adaptive_pool = report.get("semantics", {}).get("adaptive_avg_pool2d", {})
if (
    adaptive_pool.get("layout") != "NCHW"
    or adaptive_pool.get("output_shape") != [1, 1, 2, 2]
    or adaptive_pool.get("output_values") != [1.0, 2.0, 3.0, 4.0]
    or adaptive_pool.get("exact_mse") != 0.0
    or adaptive_pool.get("source_unchanged") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_adaptive_avg_pool2d]: invalid adaptive-pooling semantics"
    )
vision_classifier = report.get("semantics", {}).get("vision_classifier", {})
if (
    vision_classifier.get("pipeline")
    != ["conv2d", "relu", "adaptive_avg_pool2d", "flatten", "matmul", "add", "softmax", "argmax"]
    or vision_classifier.get("feature_shape") != [1, 1]
    or vision_classifier.get("probability_shape") != [1, 2]
    or vision_classifier.get("probability_exact_mse") != 0.0
    or vision_classifier.get("prediction_dtype") != 4
    or vision_classifier.get("prediction") != 0
    or vision_classifier.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_vision_classifier]: invalid vision-classifier semantics"
    )
batched_vision = report.get("semantics", {}).get("batched_vision_classifier", {})
if (
    batched_vision.get("pipeline")
    != ["conv2d", "relu", "adaptive_avg_pool2d", "flatten_features", "classifier_matmul", "classifier_bias", "softmax", "argmax", "cross_entropy_loss"]
    or batched_vision.get("input_shape") != [2, 1, 3, 3]
    or batched_vision.get("input_image_values") != [1.0, 2.0]
    or batched_vision.get("convolution_shape") != [2, 1, 2, 2]
    or batched_vision.get("convolution_values")
    != [4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0]
    or batched_vision.get("convolution_exact_mse") != 0.0
    or batched_vision.get("pool_shape") != [2, 1, 1, 1]
    or batched_vision.get("pool_values") != [4.0, 8.0]
    or batched_vision.get("pool_exact_mse") != 0.0
    or batched_vision.get("feature_shape") != [2, 1]
    or batched_vision.get("probability_shape") != [2, 2]
    or batched_vision.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or batched_vision.get("probability_exact_mse") != 0.0
    or batched_vision.get("prediction_shape") != [2]
    or batched_vision.get("prediction_dtype") != 4
    or batched_vision.get("predictions") != [0, 0]
    or batched_vision.get("class_targets") != [0, 0]
    or batched_vision.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_batched_vision]: invalid batched-vision semantics"
    )
grouped_cnn = report.get("semantics", {}).get("depthwise_pointwise_cnn", {})
if (
    grouped_cnn.get("pipeline")
    != ["depthwise_conv2d", "relu", "pointwise_conv2d", "relu", "adaptive_avg_pool2d", "flatten_features", "classifier_matmul", "classifier_bias", "softmax", "argmax", "cross_entropy_loss"]
    or grouped_cnn.get("input_shape") != [2, 2, 3, 3]
    or grouped_cnn.get("depthwise_groups") != 2
    or grouped_cnn.get("depthwise_weight_shape") != [2, 1, 2, 2]
    or grouped_cnn.get("depthwise_shape") != [2, 2, 2, 2]
    or grouped_cnn.get("depthwise_values") != [4.0, 8.0]
    or grouped_cnn.get("depthwise_exact_mse") != 0.0
    or grouped_cnn.get("pointwise_groups") != 1
    or grouped_cnn.get("pointwise_weight_shape") != [2, 2, 1, 1]
    or grouped_cnn.get("pointwise_shape") != [2, 2, 2, 2]
    or grouped_cnn.get("pointwise_values") != [8.0, 16.0]
    or grouped_cnn.get("pointwise_exact_mse") != 0.0
    or grouped_cnn.get("pool_shape") != [2, 2, 1, 1]
    or grouped_cnn.get("pool_values") != [8.0, 8.0, 16.0, 16.0]
    or grouped_cnn.get("pool_exact_mse") != 0.0
    or grouped_cnn.get("feature_shape") != [2, 2]
    or grouped_cnn.get("probability_shape") != [2, 2]
    or grouped_cnn.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or grouped_cnn.get("probability_exact_mse") != 0.0
    or grouped_cnn.get("prediction_shape") != [2]
    or grouped_cnn.get("prediction_dtype") != 4
    or grouped_cnn.get("predictions") != [0, 0]
    or grouped_cnn.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_grouped_cnn]: invalid grouped-CNN semantics"
    )
residual_cnn = report.get("semantics", {}).get("normalized_residual_cnn", {})
if (
    residual_cnn.get("pipeline")
    != ["depthwise_conv2d", "batch_norm", "relu", "pointwise_conv2d", "batch_norm", "residual_add", "relu", "adaptive_avg_pool2d", "flatten_features", "classifier_matmul", "classifier_bias", "softmax", "argmax", "cross_entropy_loss"]
    or residual_cnn.get("input_shape") != [2, 2, 2, 2]
    or residual_cnn.get("depthwise_groups") != 2
    or residual_cnn.get("depthwise_shape") != [2, 2, 2, 2]
    or residual_cnn.get("depthwise_values") != [1.0, 2.0]
    or residual_cnn.get("depthwise_exact_mse") != 0.0
    or residual_cnn.get("depthwise_normalized_exact_mse") != 0.0
    or residual_cnn.get("pointwise_shape") != [2, 2, 2, 2]
    or residual_cnn.get("pointwise_values") != [2.0, 4.0]
    or residual_cnn.get("pointwise_normalized_exact_mse") != 0.0
    or residual_cnn.get("residual_shape") != [2, 2, 2, 2]
    or residual_cnn.get("residual_values") != [3.0, 6.0]
    or residual_cnn.get("residual_exact_mse") != 0.0
    or residual_cnn.get("pool_shape") != [2, 2, 1, 1]
    or residual_cnn.get("pool_values") != [3.0, 3.0, 6.0, 6.0]
    or residual_cnn.get("pool_exact_mse") != 0.0
    or residual_cnn.get("feature_shape") != [2, 2]
    or residual_cnn.get("probability_shape") != [2, 2]
    or residual_cnn.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or residual_cnn.get("probability_exact_mse") != 0.0
    or residual_cnn.get("prediction_shape") != [2]
    or residual_cnn.get("prediction_dtype") != 4
    or residual_cnn.get("predictions") != [0, 0]
    or residual_cnn.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_residual_cnn]: invalid normalized-residual-CNN semantics"
    )
embedding = report.get("semantics", {}).get("embedding", {})
if (
    embedding.get("weight_shape") != [3, 2]
    or embedding.get("indices") != [0, 2]
    or embedding.get("output_shape") != [2, 2]
    or embedding.get("exact_mse") != 0.0
    or embedding.get("sources_unchanged") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_embedding]: invalid embedding semantics"
    )
sequence_classifier = report.get("semantics", {}).get("sequence_classifier", {})
if (
    sequence_classifier.get("pipeline")
    != ["embedding", "amax", "matmul", "add", "softmax", "argmax"]
    or sequence_classifier.get("embedding_shape") != [1, 2, 2]
    or sequence_classifier.get("pooled_shape") != [1, 2]
    or sequence_classifier.get("probability_shape") != [1, 2]
    or sequence_classifier.get("probability_exact_mse") != 0.0
    or sequence_classifier.get("prediction_dtype") != 4
    or sequence_classifier.get("prediction") != 0
    or sequence_classifier.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_sequence_classifier]: invalid sequence-classifier semantics"
    )
batch_norm = report.get("semantics", {}).get("batch_norm", {})
if (
    batch_norm.get("mode") != "inference"
    or batch_norm.get("input_shape") != [1, 2, 2, 2]
    or batch_norm.get("output_shape") != [1, 2, 2, 2]
    or batch_norm.get("output_value") != 3.0
    or batch_norm.get("exact_mse") != 0.0
    or batch_norm.get("sources_unchanged") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_batch_norm]: invalid batch-normalization semantics"
    )
layer_norm = report.get("semantics", {}).get("layer_norm", {})
if (
    layer_norm.get("normalized_shape") != [2]
    or layer_norm.get("output_shape") != [2, 2]
    or layer_norm.get("output_values") != [-1.0, 1.0, -1.0, 1.0]
    or layer_norm.get("exact_mse") != 0.0
    or layer_norm.get("sources_unchanged") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_layer_norm]: invalid layer-normalization semantics"
    )
transformer_residual = report.get("semantics", {}).get("transformer_residual", {})
if (
    transformer_residual.get("pipeline") != ["layer_norm", "matmul", "add"]
    or transformer_residual.get("residual_shape") != [2, 2]
    or transformer_residual.get("residual_values") != [0.0, 4.0, 1.0, 5.0]
    or transformer_residual.get("normalized_exact_mse") != 0.0
    or transformer_residual.get("residual_exact_mse") != 0.0
    or transformer_residual.get("input_unchanged") is not True
    or transformer_residual.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_transformer_residual]: invalid transformer-residual semantics"
    )
self_attention = report.get("semantics", {}).get("self_attention", {})
if (
    self_attention.get("pipeline")
    != ["layer_norm", "qkv_matmul", "value_bias", "transpose", "score_matmul", "scale", "softmax", "context_matmul", "residual_add"]
    or self_attention.get("attention_shape") != [2, 2]
    or self_attention.get("attention_values") != [0.5, 0.5, 0.5, 0.5]
    or self_attention.get("context_values") != [1.0, 1.0, 1.0, 1.0]
    or self_attention.get("residual_values") != [2.0, 4.0, 5.0, 3.0]
    or self_attention.get("attention_exact_mse") != 0.0
    or self_attention.get("context_exact_mse") != 0.0
    or self_attention.get("residual_exact_mse") != 0.0
    or self_attention.get("input_unchanged") is not True
    or self_attention.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_self_attention]: invalid self-attention semantics"
    )
gelu = report.get("semantics", {}).get("gelu", {})
if (
    gelu.get("approximate") != "none"
    or gelu.get("input_shape") != [3]
    or gelu.get("output_shape") != [3]
    or gelu.get("input_unchanged") is not True
):
    raise SystemExit("error[terlan_pytorch_report_gelu]: invalid GELU semantics")
feed_forward = report.get("semantics", {}).get("feed_forward", {})
if (
    feed_forward.get("pipeline")
    != ["layer_norm", "expansion_matmul", "expansion_bias", "gelu", "projection_matmul", "projection_bias", "residual_add"]
    or feed_forward.get("expanded_shape") != [2, 4]
    or feed_forward.get("projected_shape") != [2, 2]
    or feed_forward.get("residual_shape") != [2, 2]
    or feed_forward.get("input_unchanged") is not True
    or feed_forward.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_feed_forward]: invalid feed-forward semantics"
    )
transformer_encoder = report.get("semantics", {}).get("transformer_encoder", {})
if (
    transformer_encoder.get("pipeline")
    != ["attention_layer_norm", "qkv_matmul", "scaled_dot_product_attention", "attention_residual", "feed_forward_layer_norm", "expansion_matmul", "gelu", "projection_matmul", "feed_forward_residual"]
    or transformer_encoder.get("input_shape") != [2, 2]
    or transformer_encoder.get("attention_shape") != [2, 2]
    or transformer_encoder.get("attention_residual_values") != [2.0, 4.0, 5.0, 3.0]
    or transformer_encoder.get("expanded_shape") != [2, 4]
    or transformer_encoder.get("output_shape") != [2, 2]
    or transformer_encoder.get("input_unchanged") is not True
    or transformer_encoder.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_transformer_encoder]: invalid transformer-encoder semantics"
    )
transformer_classifier = report.get("semantics", {}).get("transformer_classifier", {})
if (
    transformer_classifier.get("pipeline")
    != ["transformer_encoder", "token_amax", "unsqueeze_batch", "classifier_matmul", "classifier_bias", "softmax", "argmax", "cross_entropy_loss"]
    or transformer_classifier.get("encoder_shape") != [2, 2]
    or transformer_classifier.get("pooled_shape") != [2]
    or transformer_classifier.get("batched_shape") != [1, 2]
    or transformer_classifier.get("probability_shape") != [1, 2]
    or transformer_classifier.get("probability_values") != [0.5, 0.5]
    or transformer_classifier.get("probability_exact_mse") != 0.0
    or transformer_classifier.get("prediction_shape") != [1]
    or transformer_classifier.get("prediction_dtype") != 4
    or transformer_classifier.get("prediction") != 0
    or transformer_classifier.get("input_unchanged") is not True
    or transformer_classifier.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_transformer_classifier]: invalid transformer-classifier semantics"
    )
token_transformer = report.get("semantics", {}).get("token_transformer_classifier", {})
if (
    token_transformer.get("pipeline")
    != ["embedding", "transformer_encoder", "token_amax", "unsqueeze_batch", "classifier_matmul", "classifier_bias", "softmax", "argmax", "cross_entropy_loss"]
    or token_transformer.get("token_ids") != [0, 1]
    or token_transformer.get("embedding_table_shape") != [2, 2]
    or token_transformer.get("embedding_shape") != [2, 2]
    or token_transformer.get("embedding_values") != [1.0, 3.0, 4.0, 2.0]
    or token_transformer.get("embedding_exact_mse") != 0.0
    or token_transformer.get("encoder_shape") != [2, 2]
    or token_transformer.get("pooled_shape") != [2]
    or token_transformer.get("batched_shape") != [1, 2]
    or token_transformer.get("probability_shape") != [1, 2]
    or token_transformer.get("probability_values") != [0.5, 0.5]
    or token_transformer.get("probability_exact_mse") != 0.0
    or token_transformer.get("prediction_shape") != [1]
    or token_transformer.get("prediction_dtype") != 4
    or token_transformer.get("prediction") != 0
    or token_transformer.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_token_transformer]: invalid token-transformer semantics"
    )
batched_transformer = report.get("semantics", {}).get("batched_token_transformer", {})
if (
    batched_transformer.get("pipeline")
    != ["embedding", "batched_transformer_encoder", "token_amax", "classifier_matmul", "classifier_bias", "softmax", "argmax", "cross_entropy_loss"]
    or batched_transformer.get("token_ids") != [[0, 1], [1, 0]]
    or batched_transformer.get("embedding_table_shape") != [2, 2]
    or batched_transformer.get("embedding_shape") != [2, 2, 2]
    or batched_transformer.get("embedding_exact_mse") != 0.0
    or batched_transformer.get("attention_key_transpose_dimensions") != [1, 2]
    or batched_transformer.get("attention_softmax_dimension") != 2
    or batched_transformer.get("encoder_shape") != [2, 2, 2]
    or batched_transformer.get("token_pool_dimension") != 1
    or batched_transformer.get("pooled_shape") != [2, 2]
    or batched_transformer.get("probability_shape") != [2, 2]
    or batched_transformer.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or batched_transformer.get("probability_exact_mse") != 0.0
    or batched_transformer.get("prediction_shape") != [2]
    or batched_transformer.get("prediction_dtype") != 4
    or batched_transformer.get("predictions") != [0, 0]
    or batched_transformer.get("class_targets") != [0, 0]
    or batched_transformer.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_batched_transformer]: invalid batched-transformer semantics"
    )
stacked_transformer = report.get("semantics", {}).get("stacked_batched_transformer", {})
if (
    stacked_transformer.get("pipeline")
    != ["embedding", "transformer_encoder_layer_1", "transformer_encoder_layer_2", "token_amax", "classifier_matmul", "classifier_bias", "softmax", "argmax", "cross_entropy_loss"]
    or stacked_transformer.get("token_ids") != [[0, 1], [1, 0]]
    or stacked_transformer.get("embedding_shape") != [2, 2, 2]
    or stacked_transformer.get("layer_count") != 2
    or stacked_transformer.get("first_encoder_shape") != [2, 2, 2]
    or stacked_transformer.get("second_encoder_shape") != [2, 2, 2]
    or stacked_transformer.get("second_encoder_values")
    != [2.6826894921, 6.6826894921, 7.6826894921, 3.6826894921, 7.6826894921, 3.6826894921, 2.6826894921, 6.6826894921]
    or stacked_transformer.get("pooled_shape") != [2, 2]
    or stacked_transformer.get("pooled_values")
    != [7.6826894921, 6.6826894921, 7.6826894921, 6.6826894921]
    or stacked_transformer.get("probability_shape") != [2, 2]
    or stacked_transformer.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or stacked_transformer.get("probability_exact_mse") != 0.0
    or stacked_transformer.get("prediction_shape") != [2]
    or stacked_transformer.get("prediction_dtype") != 4
    or stacked_transformer.get("predictions") != [0, 0]
    or stacked_transformer.get("parameters_reused_across_layers") is not True
    or stacked_transformer.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_stacked_transformer]: invalid stacked-transformer semantics"
    )
causal_transformer = report.get("semantics", {}).get("causal_batched_transformer", {})
if (
    causal_transformer.get("pipeline")
    != ["embedding", "qkv_matmul", "scaled_scores", "additive_causal_mask", "softmax", "context_matmul", "attention_residual", "feed_forward", "token_amax", "classifier_matmul", "softmax", "argmax", "cross_entropy_loss"]
    or causal_transformer.get("token_ids") != [[0, 1], [1, 0]]
    or causal_transformer.get("embedding_shape") != [2, 2, 2]
    or causal_transformer.get("mask_shape") != [2, 2]
    or causal_transformer.get("mask_values") != [0.0, -1000.0, 0.0, 0.0]
    or causal_transformer.get("mask_broadcast_over_batch") is not True
    or causal_transformer.get("attention_shape") != [2, 2, 2]
    or causal_transformer.get("attention_values")
    != [1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5]
    or causal_transformer.get("attention_exact_mse") != 0.0
    or causal_transformer.get("context_values")
    != [0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0]
    or causal_transformer.get("context_exact_mse") != 0.0
    or causal_transformer.get("attention_residual_values")
    != [1.0, 5.0, 5.0, 3.0, 6.0, 2.0, 2.0, 4.0]
    or causal_transformer.get("attention_residual_exact_mse") != 0.0
    or causal_transformer.get("encoder_shape") != [2, 2, 2]
    or causal_transformer.get("pooled_shape") != [2, 2]
    or causal_transformer.get("pooled_values")
    != [5.8413447461, 5.8413447461, 6.8413447461, 4.8413447461]
    or causal_transformer.get("probability_shape") != [2, 2]
    or causal_transformer.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or causal_transformer.get("prediction_shape") != [2]
    or causal_transformer.get("prediction_dtype") != 4
    or causal_transformer.get("predictions") != [0, 0]
    or causal_transformer.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_causal_transformer]: invalid causal-transformer semantics"
    )

cross_attention = report.get("semantics", {}).get("cross_attention_transformer", {})
if (
    cross_attention.get("pipeline")
    != ["decoder_layer_norm", "memory_layer_norm", "separate_qkv_matmul", "memory_key_transpose", "scaled_cross_attention", "softmax", "memory_context_matmul", "decoder_residual", "feed_forward", "token_amax", "classifier_matmul", "softmax", "argmax", "cross_entropy_loss"]
    or cross_attention.get("decoder_shape") != [2, 2, 2]
    or cross_attention.get("memory_shape") != [2, 3, 2]
    or cross_attention.get("query_length") != 2
    or cross_attention.get("memory_length") != 3
    or cross_attention.get("attention_shape") != [2, 2, 3]
    or cross_attention.get("attention_values") != [0.3333333333333333] * 12
    or cross_attention.get("attention_mse_maximum") != 0.0000001
    or cross_attention.get("context_shape") != [2, 2, 2]
    or cross_attention.get("context_values")
    != [-0.3333333333333333, 0.3333333333333333, -0.3333333333333333, 0.3333333333333333, 0.3333333333333333, -0.3333333333333333, 0.3333333333333333, -0.3333333333333333]
    or cross_attention.get("context_mse_maximum") != 0.0000001
    or cross_attention.get("attention_residual_values")
    != [0.6666666666666667, 3.3333333333333335, 1.6666666666666667, 5.333333333333333, 3.3333333333333335, 0.6666666666666667, 5.333333333333333, 1.6666666666666667]
    or cross_attention.get("attention_residual_mse_maximum") != 0.0000001
    or cross_attention.get("encoder_shape") != [2, 2, 2]
    or cross_attention.get("encoder_values")
    != [0.5080114128, 4.1746780794, 1.5080114128, 6.1746780794, 4.1746780794, 0.5080114128, 6.1746780794, 1.5080114128]
    or cross_attention.get("encoder_mse_maximum") != 0.0000001
    or cross_attention.get("pooled_shape") != [2, 2]
    or cross_attention.get("pooled_values")
    != [1.5080114128, 6.1746780794, 6.1746780794, 1.5080114128]
    or cross_attention.get("pooled_mse_maximum") != 0.0000001
    or cross_attention.get("probability_shape") != [2, 2]
    or cross_attention.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or cross_attention.get("prediction_shape") != [2]
    or cross_attention.get("prediction_dtype") != 4
    or cross_attention.get("predictions") != [0, 0]
    or cross_attention.get("cross_entropy_minimum") != 0.69
    or cross_attention.get("cross_entropy_maximum") != 0.70
    or cross_attention.get("decoder_and_memory_independently_owned") is not True
    or cross_attention.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_cross_attention]: invalid cross-attention transformer semantics"
    )

decoder = report.get("semantics", {}).get("causal_cross_attention_decoder", {})
if (
    decoder.get("pipeline")
    != ["embedding", "causal_self_attention", "self_attention_residual", "memory_cross_attention", "cross_attention_residual", "feed_forward", "token_amax", "classifier_matmul", "softmax", "argmax", "cross_entropy_loss"]
    or decoder.get("decoder_token_ids") != [[0, 1], [1, 0]]
    or decoder.get("decoder_input_shape") != [2, 2, 2]
    or decoder.get("causal_mask_shape") != [2, 2]
    or decoder.get("memory_shape") != [2, 3, 2]
    or decoder.get("self_attention_residual_shape") != [2, 2, 2]
    or decoder.get("self_attention_residual_values")
    != [1.0, 5.0, 5.0, 3.0, 6.0, 2.0, 2.0, 4.0]
    or decoder.get("self_attention_residual_exact_mse") != 0.0
    or decoder.get("cross_attention_shape") != [2, 2, 3]
    or decoder.get("cross_attention_residual_shape") != [2, 2, 2]
    or decoder.get("cross_attention_residual_values")
    != [0.6666666666666667, 5.333333333333333, 4.666666666666667, 3.3333333333333335, 6.333333333333333, 1.6666666666666667, 2.3333333333333335, 3.6666666666666665]
    or decoder.get("cross_attention_residual_mse_maximum") != 0.0000001
    or decoder.get("decoder_output_shape") != [2, 2, 2]
    or decoder.get("decoder_output_values")
    != [0.5080114128, 6.1746780794, 5.5080114128, 3.1746780794, 7.1746780794, 1.5080114128, 2.1746780794, 4.5080114128]
    or decoder.get("decoder_output_mse_maximum") != 0.0000001
    or decoder.get("pooled_shape") != [2, 2]
    or decoder.get("pooled_values")
    != [5.5080114128, 6.1746780794, 7.1746780794, 4.5080114128]
    or decoder.get("pooled_mse_maximum") != 0.0000001
    or decoder.get("probability_shape") != [2, 2]
    or decoder.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or decoder.get("prediction_shape") != [2]
    or decoder.get("prediction_dtype") != 4
    or decoder.get("predictions") != [0, 0]
    or decoder.get("cross_entropy_minimum") != 0.69
    or decoder.get("cross_entropy_maximum") != 0.70
    or decoder.get("causal_and_cross_attention_composed") is not True
    or decoder.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_decoder]: invalid causal cross-attention decoder semantics"
    )

seq2seq = report.get("semantics", {}).get("tokenized_seq2seq", {})
if (
    seq2seq.get("pipeline")
    != ["source_token_ids", "source_embedding", "encoder_memory", "target_token_ids", "target_embedding", "causal_self_attention", "memory_cross_attention", "feed_forward", "token_amax", "classifier_matmul", "softmax", "argmax", "cross_entropy_loss"]
    or seq2seq.get("source_token_ids") != [[0, 0, 1], [1, 1, 0]]
    or seq2seq.get("source_embedding_table_shape") != [2, 2]
    or seq2seq.get("memory_shape") != [2, 3, 2]
    or seq2seq.get("memory_values")
    != [0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0]
    or seq2seq.get("memory_exact_mse") != 0.0
    or seq2seq.get("target_token_ids") != [[0, 1], [1, 0]]
    or seq2seq.get("target_embedding_table_shape") != [2, 2]
    or seq2seq.get("decoder_input_shape") != [2, 2, 2]
    or seq2seq.get("decoder_input_values")
    != [1.0, 3.0, 4.0, 2.0, 4.0, 2.0, 1.0, 3.0]
    or seq2seq.get("decoder_input_exact_mse") != 0.0
    or seq2seq.get("separate_source_and_target_embeddings") is not True
    or seq2seq.get("decoder_output_shape") != [2, 2, 2]
    or seq2seq.get("decoder_output_values")
    != [0.5080114128, 6.1746780794, 5.5080114128, 3.1746780794, 7.1746780794, 1.5080114128, 2.1746780794, 4.5080114128]
    or seq2seq.get("decoder_output_mse_maximum") != 0.0000001
    or seq2seq.get("pooled_shape") != [2, 2]
    or seq2seq.get("pooled_values")
    != [5.5080114128, 6.1746780794, 7.1746780794, 4.5080114128]
    or seq2seq.get("pooled_mse_maximum") != 0.0000001
    or seq2seq.get("probability_shape") != [2, 2]
    or seq2seq.get("probability_values") != [0.5, 0.5, 0.5, 0.5]
    or seq2seq.get("prediction_shape") != [2]
    or seq2seq.get("prediction_dtype") != 4
    or seq2seq.get("predictions") != [0, 0]
    or seq2seq.get("cross_entropy_minimum") != 0.69
    or seq2seq.get("cross_entropy_maximum") != 0.70
    or seq2seq.get("all_handles_disposed") is not True
):
    raise SystemExit(
        "error[terlan_pytorch_report_seq2seq]: invalid tokenized seq2seq semantics"
    )

print(f"validated passing Terlan PyTorch report: {path}")
