#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use terlan_pytorch_native::{CAbiError, Tensor, TensorPair, TorchScriptModel, abi_version};

fn main() {
    let mut worker = Worker::default();
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let payload = match line {
            Ok(line) => worker.execute_line(&line),
            Err(error) => protocol_error("native_read_error", &error.to_string()),
        };
        if writeln!(stdout, "{payload}").is_err() || stdout.flush().is_err() {
            break;
        }
    }
}

#[derive(Default)]
struct Worker {
    next_id: u64,
    handles: HashMap<u64, HandleEntry>,
}

struct HandleEntry {
    generation: u64,
    value: HandleValue,
}

enum HandleValue {
    Tensor(Tensor),
    TorchScriptModel(TorchScriptModel),
    TensorPair(TensorPair),
}

impl HandleValue {
    fn type_name(&self) -> &'static str {
        match self {
            HandleValue::Tensor(_) => "pytorch.Tensor.Tensor",
            HandleValue::TorchScriptModel(_) => "pytorch.Tensor.TorchScriptModel",
            HandleValue::TensorPair(_) => "pytorch.Tensor.TensorPair",
        }
    }
}

#[derive(Clone)]
struct HandleArg {
    id: u64,
    generation: u64,
    type_name: String,
}

enum Arg {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    EmptyList,
    Handle(HandleArg),
}

struct Request {
    request_id: u64,
    operation: String,
    args: Vec<Arg>,
}

impl Worker {
    fn execute_line(&mut self, line: &str) -> String {
        let request = match parse_request(line) {
            Ok(request) => request,
            Err(error) => {
                return request_id(line)
                    .map(|request_id| format!("reply {request_id} 1 {error}"))
                    .unwrap_or(error);
            }
        };
        let request_id = request.request_id;
        let payload = self.execute(request);
        format!("reply {request_id} 1 {payload}")
    }

    fn execute(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "pytorch.tensor.from_int" => {
                let [Arg::Int(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_int expects its declared arguments");
                };
                let value = match Tensor::from_int(*value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.from_float" => {
                let [Arg::Float(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_float expects its declared arguments");
                };
                let value = match Tensor::from_float(*value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.from_bool" => {
                let [Arg::Bool(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_bool expects its declared arguments");
                };
                let value = match Tensor::from_bool(*value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.from_floats" => {
                let [values @ (Arg::Floats(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_floats expects its declared arguments");
                };
                let value = match Tensor::from_floats(arg_floats(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.from_ints" => {
                let [values @ (Arg::Ints(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_ints expects its declared arguments");
                };
                let value = match Tensor::from_ints(arg_ints(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.full_float" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "full_float expects its declared arguments");
                };
                let value = match Tensor::full_float(arg_ints(shape), *value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.item" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "item expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.item() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.item_float" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "item_float expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.item_float() { Ok(value) => format!("ok_float {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.item_bool" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "item_bool expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.item_bool() { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.dim" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dim expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.dim() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.numel" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "numel expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.numel() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.shape" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "shape expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.shape() { Ok(values) if values.is_empty() => "ok_ints".to_string(), Ok(values) => format!("ok_ints {}", values.iter().map(i64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.dtype" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dtype expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.dtype() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.device_type" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "device_type expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.device_type() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.device_index" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "device_index expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_tensor.device_index() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.tensor.clone" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "clone expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.clone() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.unsqueeze" => {
                let [Arg::Handle(tensor), Arg::Int(dimension)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unsqueeze expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.unsqueeze(*dimension) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.matmul" => {
                let [Arg::Handle(left), Arg::Handle(right)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "matmul expects its declared arguments");
                };
                let value_left = match self.live_tensor(left) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_right = match self.live_tensor(right) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_left.matmul(value_right) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.transpose" => {
                let [Arg::Handle(tensor), Arg::Int(dimension0), Arg::Int(dimension1)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "transpose expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.transpose(*dimension0, *dimension1) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.multiply" => {
                let [Arg::Handle(left), Arg::Handle(right)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "multiply expects its declared arguments");
                };
                let value_left = match self.live_tensor(left) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_right = match self.live_tensor(right) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_left.multiply(value_right) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.add" => {
                let [Arg::Handle(left), Arg::Handle(right), Arg::Float(alpha)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "add expects its declared arguments");
                };
                let value_left = match self.live_tensor(left) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_right = match self.live_tensor(right) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_left.add(value_right, *alpha) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.subtract" => {
                let [Arg::Handle(left), Arg::Handle(right), Arg::Float(alpha)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "subtract expects its declared arguments");
                };
                let value_left = match self.live_tensor(left) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_right = match self.live_tensor(right) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_left.subtract(value_right, *alpha) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.relu" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "relu expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.relu() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.gelu" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gelu expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.gelu() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.softmax" => {
                let [Arg::Handle(tensor), Arg::Int(dimension)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "softmax expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.softmax(*dimension) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.argmax" => {
                let [Arg::Handle(tensor), Arg::Int(dimension), Arg::Bool(keep_dimension)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argmax expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.argmax(*dimension, *keep_dimension) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.embedding" => {
                let [Arg::Handle(weight), Arg::Handle(indices), Arg::Int(padding_index), Arg::Bool(scale_gradient_by_frequency), Arg::Bool(sparse)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "embedding expects its declared arguments");
                };
                let value_weight = match self.live_tensor(weight) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_indices = match self.live_tensor(indices) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_weight.embedding(value_indices, *padding_index, *scale_gradient_by_frequency, *sparse) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.batch_norm" => {
                let [Arg::Handle(input), Arg::Handle(weight), Arg::Handle(bias), Arg::Handle(running_mean), Arg::Handle(running_variance), Arg::Bool(training), Arg::Float(momentum), Arg::Float(epsilon), Arg::Bool(cudnn_enabled)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "batch_norm expects its declared arguments");
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weight = match self.live_tensor(weight) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_bias = match self.live_tensor(bias) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_running_mean = match self.live_tensor(running_mean) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_running_variance = match self.live_tensor(running_variance) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_input.batch_norm(value_weight, value_bias, value_running_mean, value_running_variance, *training, *momentum, *epsilon, *cudnn_enabled) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.layer_norm" => {
                let [Arg::Handle(input), normalized_shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(weight), Arg::Handle(bias), Arg::Float(epsilon), Arg::Bool(cudnn_enabled)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "layer_norm expects its declared arguments");
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weight = match self.live_tensor(weight) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_bias = match self.live_tensor(bias) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_input.layer_norm(arg_ints(normalized_shape), value_weight, value_bias, *epsilon, *cudnn_enabled) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.mse_loss" => {
                let [Arg::Handle(predicted), Arg::Handle(target), Arg::Int(reduction)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mse_loss expects its declared arguments");
                };
                let value_predicted = match self.live_tensor(predicted) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_target = match self.live_tensor(target) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_predicted.mse_loss(value_target, *reduction) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.mse_loss_gradient" => {
                let [Arg::Handle(grad_output), Arg::Handle(predicted), Arg::Handle(target), Arg::Int(reduction)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mse_loss_gradient expects its declared arguments");
                };
                let value_grad_output = match self.live_tensor(grad_output) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_predicted = match self.live_tensor(predicted) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_target = match self.live_tensor(target) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_grad_output.mse_loss_gradient(value_predicted, value_target, *reduction) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.cross_entropy_loss" => {
                let [Arg::Handle(logits), Arg::Handle(target), Arg::Int(reduction), Arg::Int(ignore_index), Arg::Float(label_smoothing)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "cross_entropy_loss expects its declared arguments");
                };
                let value_logits = match self.live_tensor(logits) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_target = match self.live_tensor(target) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_logits.cross_entropy_loss(value_target, *reduction, *ignore_index, *label_smoothing) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.narrow" => {
                let [Arg::Handle(tensor), Arg::Int(dimension), Arg::Int(start), Arg::Int(length)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "narrow expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.narrow(*dimension, *start, *length) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.flatten" => {
                let [Arg::Handle(tensor), Arg::Int(start_dimension), Arg::Int(end_dimension)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "flatten expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.flatten(*start_dimension, *end_dimension) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.amax" => {
                let [Arg::Handle(tensor), dimensions @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dimension)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "amax expects its declared arguments");
                };
                let value_tensor = match self.live_tensor(tensor) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_tensor.amax(arg_ints(dimensions), *keep_dimension) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.conv2d" => {
                let [Arg::Handle(input), Arg::Handle(weight), Arg::Handle(bias), stride @ (Arg::Ints(_) | Arg::EmptyList), padding @ (Arg::Ints(_) | Arg::EmptyList), dilation @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(groups)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "conv2d expects its declared arguments");
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weight = match self.live_tensor(weight) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_bias = match self.live_tensor(bias) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_input.conv2d(value_weight, value_bias, arg_ints(stride), arg_ints(padding), arg_ints(dilation), *groups) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.adaptive_avg_pool2d" => {
                let [Arg::Handle(input), output_size @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "adaptive_avg_pool2d expects its declared arguments");
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_input.adaptive_avg_pool2d(arg_ints(output_size)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.torchscript.load" => {
                let [Arg::String(path)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "load_torchscript expects its declared arguments");
                };
                let value = match TorchScriptModel::load_torchscript(path.as_str()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::TorchScriptModel(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.TorchScriptModel"))
            }
            "pytorch.torchscript.forward" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_model.torchscript_forward(value_input) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.torchscript.forward2" => {
                let [Arg::Handle(model), Arg::Handle(first), Arg::Handle(second)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward2 expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_first = match self.live_tensor(first) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_second = match self.live_tensor(second) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_model.torchscript_forward2(value_first, value_second) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.torchscript.forward_pair" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_pair expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_model.torchscript_forward_pair(value_input) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::TensorPair(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.TensorPair"))
            }
            "pytorch.torchscript.forward_float" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_float expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_float(value_input) { Ok(value) => format!("ok_float {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.forward_int" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_int expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_int(value_input) { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.forward_bool" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_bool expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_bool(value_input) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.forward_string" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_string expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_string(value_input) { Ok(value) => format!("ok_string {}", STANDARD.encode(value.as_bytes())), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.forward_int_list" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_int_list expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_int_list(value_input) { Ok(values) if values.is_empty() => "ok_ints".to_string(), Ok(values) => format!("ok_ints {}", values.iter().map(i64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.forward_float_list" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_float_list expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_float_list(value_input) { Ok(values) if values.is_empty() => "ok_floats".to_string(), Ok(values) => format!("ok_floats {}", values.iter().map(f64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.forward_bool_list" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_bool_list expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_bool_list(value_input) { Ok(values) if values.is_empty() => "ok_bools".to_string(), Ok(values) => format!("ok_bools {}", values.iter().map(bool::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.forward_string_list" => {
                let [Arg::Handle(model), Arg::Handle(input)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "torchscript_forward_string_list expects its declared arguments");
                };
                let value_model = match self.live_torchscriptmodel(model) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_model.torchscript_forward_string_list(value_input) { Ok(values) if values.is_empty() => "ok_strings".to_string(), Ok(values) => format!("ok_strings {}", values.iter().map(|value| STANDARD.encode(value.as_bytes())).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "pytorch.torchscript.tensor_pair_first" => {
                let [Arg::Handle(pair)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tensor_pair_first expects its declared arguments");
                };
                let value_pair = match self.live_tensorpair(pair) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_pair.tensor_pair_first() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.torchscript.tensor_pair_second" => {
                let [Arg::Handle(pair)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tensor_pair_second expects its declared arguments");
                };
                let value_pair = match self.live_tensorpair(pair) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_pair.tensor_pair_second() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.tensor.run_torchscript" => {
                let [Arg::Handle(input), Arg::String(path)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "run_torchscript expects its declared arguments");
                };
                let value_input = match self.live_tensor(input) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_input.run_torchscript(path.as_str()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Tensor(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("pytorch.Tensor.Tensor"))
            }
            "pytorch.torchscript.dispose" => {
                let [Arg::Handle(model)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dispose_model expects its declared arguments");
                };
                if let Err(error) = self.validate(model, "pytorch.Tensor.TorchScriptModel") {
                    return error;
                }
                self.handles.remove(&model.id);
                "ok_unit".to_string()
            }
            "pytorch.torchscript.dispose_tensor_pair" => {
                let [Arg::Handle(pair)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dispose_tensor_pair expects its declared arguments");
                };
                if let Err(error) = self.validate(pair, "pytorch.Tensor.TensorPair") {
                    return error;
                }
                self.handles.remove(&pair.id);
                "ok_unit".to_string()
            }
            "pytorch.tensor.dispose" => {
                let [Arg::Handle(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dispose expects its declared arguments");
                };
                if let Err(error) = self.validate(tensor, "pytorch.Tensor.Tensor") {
                    return error;
                }
                self.handles.remove(&tensor.id);
                "ok_unit".to_string()
            }
            "pytorch.abi_version" => {
                let [] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "abi_version expects its declared arguments");
                };
                format!("ok_int {}", abi_version())
            }

            _ => protocol_error("unknown_operation", &request.operation),
        }
    }

    fn validate(&self, handle: &HandleArg, expected_type: &str) -> Result<(), String> {
        if handle.type_name != expected_type {
            return Err(protocol_error("handle_type_mismatch", &handle.type_name));
        }
        match self.handles.get(&handle.id) {
            Some(entry)
                if entry.generation == handle.generation
                    && entry.value.type_name() == expected_type => Ok(()),
            Some(_) => Err(protocol_error(
                "handle_storage_mismatch",
                "NativeBoundary handle resource type does not match",
            )),
            _ => Err(protocol_error("stale_handle", "NativeBoundary handle is stale")),
        }
    }

    fn live_tensor(&self, handle: &HandleArg) -> Result<&Tensor, String> {
        self.validate(handle, "pytorch.Tensor.Tensor")?;
        match &self.handles.get(&handle.id).expect("validated handle").value {
            HandleValue::Tensor(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "pytorch.Tensor.Tensor")),
        }
    }

    #[allow(dead_code)]
    fn live_tensor_mut(&mut self, handle: &HandleArg) -> Result<&mut Tensor, String> {
        self.validate(handle, "pytorch.Tensor.Tensor")?;
        match &mut self.handles.get_mut(&handle.id).expect("validated handle").value {
            HandleValue::Tensor(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "pytorch.Tensor.Tensor")),
        }
    }

    fn live_torchscriptmodel(&self, handle: &HandleArg) -> Result<&TorchScriptModel, String> {
        self.validate(handle, "pytorch.Tensor.TorchScriptModel")?;
        match &self.handles.get(&handle.id).expect("validated handle").value {
            HandleValue::TorchScriptModel(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "pytorch.Tensor.TorchScriptModel")),
        }
    }

    #[allow(dead_code)]
    fn live_torchscriptmodel_mut(&mut self, handle: &HandleArg) -> Result<&mut TorchScriptModel, String> {
        self.validate(handle, "pytorch.Tensor.TorchScriptModel")?;
        match &mut self.handles.get_mut(&handle.id).expect("validated handle").value {
            HandleValue::TorchScriptModel(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "pytorch.Tensor.TorchScriptModel")),
        }
    }

    fn live_tensorpair(&self, handle: &HandleArg) -> Result<&TensorPair, String> {
        self.validate(handle, "pytorch.Tensor.TensorPair")?;
        match &self.handles.get(&handle.id).expect("validated handle").value {
            HandleValue::TensorPair(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "pytorch.Tensor.TensorPair")),
        }
    }

    #[allow(dead_code)]
    fn live_tensorpair_mut(&mut self, handle: &HandleArg) -> Result<&mut TensorPair, String> {
        self.validate(handle, "pytorch.Tensor.TensorPair")?;
        match &mut self.handles.get_mut(&handle.id).expect("validated handle").value {
            HandleValue::TensorPair(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "pytorch.Tensor.TensorPair")),
        }
    }


}

fn request_id(line: &str) -> Option<u64> {
    let mut fields = line.split_whitespace();
    (fields.next() == Some("call"))
        .then(|| fields.next()?.parse::<u64>().ok())
        .flatten()
}

fn parse_request(line: &str) -> Result<Request, String> {
    let mut fields = line.split_whitespace();
    if fields.next() != Some("call") {
        return Err(protocol_error("invalid_request", "expected call request"));
    }
    let request_id = fields
        .next()
        .ok_or_else(|| protocol_error("invalid_request", "missing request id"))?
        .parse::<u64>()
        .map_err(|error| protocol_error("invalid_request", &error.to_string()))?;
    let operation = decode_text(fields.next().ok_or_else(|| {
        protocol_error("invalid_request", "missing encoded operation")
    })?)?;
    let args = fields.map(parse_arg).collect::<Result<Vec<_>, _>>()?;
    for argument in &args {
        validate_decoded_arg_shape(argument);
    }
    Ok(Request { request_id, operation, args })
}

fn parse_arg(value: &str) -> Result<Arg, String> {
    // The VM uses `ls:` for an empty list. Preserve the empty value here and
    // resolve it against each generated operation's declared list type.
    if value == "ls:" {
        return Ok(Arg::EmptyList);
    }
    if let Some(value) = value.strip_prefix("i:") {
        return value
            .parse::<i64>()
            .map(Arg::Int)
            .map_err(|error| protocol_error("invalid_argument", &error.to_string()));
    }
    if let Some(value) = value.strip_prefix("f:") {
        return value
            .parse::<f64>()
            .map(Arg::Float)
            .map_err(|error| protocol_error("invalid_argument", &error.to_string()));
    }
    if let Some(value) = value.strip_prefix("b:") {
        return value
            .parse::<bool>()
            .map(Arg::Bool)
            .map_err(|error| protocol_error("invalid_argument", &error.to_string()));
    }
    if let Some(value) = value.strip_prefix("s:") {
        return decode_text(value).map(Arg::String);
    }
    if let Some(value) = value.strip_prefix("li:") {
        if value.is_empty() {
            return Ok(Arg::Ints(Vec::new()));
        }
        return value
            .split(',')
            .map(|value| {
                value.parse::<i64>().map_err(|error| {
                    protocol_error("invalid_argument", &error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Arg::Ints);
    }
    if let Some(value) = value.strip_prefix("lf:") {
        if value.is_empty() {
            return Ok(Arg::Floats(Vec::new()));
        }
        return value
            .split(',')
            .map(|value| {
                value.parse::<f64>().map_err(|error| {
                    protocol_error("invalid_argument", &error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Arg::Floats);
    }
    if let Some(value) = value.strip_prefix("h:") {
        let fields = value.split(':').collect::<Vec<_>>();
        let [id, generation, type_name] = fields.as_slice() else {
            return Err(protocol_error("invalid_argument", "malformed handle"));
        };
        return Ok(Arg::Handle(HandleArg {
            id: id.parse().map_err(|error: std::num::ParseIntError| protocol_error("invalid_argument", &error.to_string()))?,
            generation: generation.parse().map_err(|error: std::num::ParseIntError| protocol_error("invalid_argument", &error.to_string()))?,
            type_name: decode_text(type_name)?,
        }));
    }
    Err(protocol_error("invalid_argument", "unsupported argument encoding"))
}

fn decode_text(value: &str) -> Result<String, String> {
    let bytes = STANDARD
        .decode(value)
        .map_err(|error| protocol_error("invalid_base64", &error.to_string()))?;
    String::from_utf8(bytes)
        .map_err(|error| protocol_error("invalid_utf8", &error.to_string()))
}

fn arg_ints(value: &Arg) -> &[i64] {
    match value {
        Arg::Ints(values) => values.as_slice(),
        Arg::EmptyList => &[],
        _ => unreachable!("generated argument pattern admits only integer lists"),
    }
}

fn arg_floats(value: &Arg) -> &[f64] {
    match value {
        Arg::Floats(values) => values.as_slice(),
        Arg::EmptyList => &[],
        _ => unreachable!("generated argument pattern admits only float lists"),
    }
}

fn validate_decoded_arg_shape(value: &Arg) {
    match value {
        Arg::Int(value) => { let _ = value; }
        Arg::Float(value) => { let _ = value; }
        Arg::Bool(value) => { let _ = value; }
        Arg::String(value) => { let _ = value; }
        Arg::Ints(_) => { let _ = arg_ints(value); }
        Arg::Floats(_) => { let _ = arg_floats(value); }
        Arg::EmptyList => {}
        Arg::Handle(value) => { let _ = value; }
    }
}

fn native_error(error: &CAbiError) -> String {
    protocol_error(
        &format!("c_abi_status_{}", error.status),
        error.operation,
    )
}

fn protocol_error(code: &str, message: &str) -> String {
    format!("err {} {}", STANDARD.encode(code), STANDARD.encode(message))
}
