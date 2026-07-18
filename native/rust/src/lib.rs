#![deny(unsafe_op_in_unsafe_fn)]

pub mod ffi {
    #[repr(C)]
    pub struct AtenTensorOpaque {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct TerlanTorchScriptModelOpaque {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct TerlanTorchScriptTensorPairOpaque {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn aoti_torch_abi_version() -> u64;
        pub fn aoti_torch_cpu__adaptive_avg_pool2d(input: *mut AtenTensorOpaque, output_size: *const i64, output_size_length: i64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn aoti_torch_cpu_add_Tensor(left: *mut AtenTensorOpaque, right: *mut AtenTensorOpaque, alpha: f64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn aoti_torch_aten_amax(tensor: *mut AtenTensorOpaque, dimensions: *const i64, dimensions_length: i64, keep_dimension: i32, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn torch_call_dispatcher(operator_name: *const std::ffi::c_char, overload_name: *const std::ffi::c_char, stack: *mut u64, extension_abi_version: u64) -> i32;
        pub fn terlan_pytorch_conv2d(input: *mut AtenTensorOpaque, weight: *mut AtenTensorOpaque, bias: *mut AtenTensorOpaque, stride: *const i64, stride_length: i64, padding: *const i64, padding_length: i64, dilation: *const i64, dilation_length: i64, groups: i64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn torch_delete_list(list: *mut ()) -> i32;
        pub fn terlan_pytorch_delete_owned_bool_array(values: *mut u8);
        pub fn terlan_pytorch_delete_owned_float_array(values: *mut f64);
        pub fn terlan_pytorch_delete_owned_int_array(values: *mut i64);
        pub fn terlan_pytorch_delete_owned_string(value: *mut std::ffi::c_char);
        pub fn terlan_pytorch_delete_owned_string_array(values: *mut *mut std::ffi::c_char, lengths: *mut usize, count: usize);
        pub fn torch_delete_stable_ivalue(value: *mut u64) -> i32;
        pub fn torch_delete_string(string: *mut ()) -> i32;
        pub fn aoti_torch_delete_tensor_object(tensor: *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_delete_tensor_pair(pair: *mut TerlanTorchScriptTensorPairOpaque) -> i32;
        pub fn terlan_pytorch_delete_torchscript_model(model: *mut TerlanTorchScriptModelOpaque) -> i32;
        pub fn aoti_torch_aten_full(shape: *const i64, shape_length: i64, value: f64, dtype: *mut i32, layout: *mut i32, device: *mut i32, device_index: i32, pin_memory: *mut i32, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn aoti_torch_get_device_index(tensor: *mut AtenTensorOpaque, device_index: *mut i32) -> i32;
        pub fn aoti_torch_get_device_type(tensor: *mut AtenTensorOpaque, device_type: *mut i32) -> i32;
        pub fn aoti_torch_get_dim(tensor: *mut AtenTensorOpaque, dim: *mut i64) -> i32;
        pub fn aoti_torch_get_dtype(tensor: *mut AtenTensorOpaque, dtype: *mut i32) -> i32;
        pub fn aoti_torch_get_numel(tensor: *mut AtenTensorOpaque, numel: *mut i64) -> i32;
        pub fn aoti_torch_get_sizes(tensor: *mut AtenTensorOpaque, sizes: *mut *mut i64) -> i32;
        pub fn aoti_torch_item_bool(tensor: *mut AtenTensorOpaque, item: *mut bool) -> i32;
        pub fn aoti_torch_item_float64(tensor: *mut AtenTensorOpaque, item: *mut f64) -> i32;
        pub fn aoti_torch_item_int64(tensor: *mut AtenTensorOpaque, item: *mut i64) -> i32;
        pub fn torch_list_push_back(list: *mut (), element: u64) -> i32;
        pub fn terlan_pytorch_load_torchscript(path: *const std::ffi::c_char, out_model: *mut *mut TerlanTorchScriptModelOpaque) -> i32;
        pub fn aoti_torch_aten_narrow(tensor: *mut AtenTensorOpaque, dimension: i64, start: i64, length: i64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn torch_new_list_reserve_size(capacity: usize, list: *mut *mut ()) -> i32;
        pub fn torch_new_stable_ivalue(value: *mut *mut u64) -> i32;
        pub fn torch_new_string_handle(data: *const std::ffi::c_char, length: usize, string: *mut *mut ()) -> i32;
        pub fn aoti_torch_new_tensor_handle(tensor: *mut AtenTensorOpaque, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_run_torchscript(input: *mut AtenTensorOpaque, path: *const std::ffi::c_char, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn aoti_torch_scalar_to_tensor_bool(value: bool, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn aoti_torch_scalar_to_tensor_float64(value: f64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn aoti_torch_scalar_to_tensor_int64(value: i64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn aoti_torch_aten_subtract_Tensor(left: *mut AtenTensorOpaque, right: *mut AtenTensorOpaque, alpha: f64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_tensor_from_f64(values: *const f64, values_length: i64, shape: *const i64, shape_length: i64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_tensor_from_i64(values: *const i64, values_length: i64, shape: *const i64, shape_length: i64, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_tensor_pair_first(pair: *mut TerlanTorchScriptTensorPairOpaque, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_tensor_pair_second(pair: *mut TerlanTorchScriptTensorPairOpaque, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_torchscript_forward(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_torchscript_forward2(model: *mut TerlanTorchScriptModelOpaque, first: *mut AtenTensorOpaque, second: *mut AtenTensorOpaque, out_tensor: *mut *mut AtenTensorOpaque) -> i32;
        pub fn terlan_pytorch_torchscript_forward_bool(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_value: *mut bool) -> i32;
        pub fn terlan_pytorch_torchscript_forward_bool_list(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_values: *mut *mut u8, out_length: *mut usize) -> i32;
        pub fn terlan_pytorch_torchscript_forward_float(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_value: *mut f64) -> i32;
        pub fn terlan_pytorch_torchscript_forward_float_list(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_values: *mut *mut f64, out_length: *mut usize) -> i32;
        pub fn terlan_pytorch_torchscript_forward_int(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_value: *mut i64) -> i32;
        pub fn terlan_pytorch_torchscript_forward_int_list(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_values: *mut *mut i64, out_length: *mut usize) -> i32;
        pub fn terlan_pytorch_torchscript_forward_pair(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_pair: *mut *mut TerlanTorchScriptTensorPairOpaque) -> i32;
        pub fn terlan_pytorch_torchscript_forward_string(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_value: *mut *mut std::ffi::c_char, out_length: *mut usize) -> i32;
        pub fn terlan_pytorch_torchscript_forward_string_list(model: *mut TerlanTorchScriptModelOpaque, input: *mut AtenTensorOpaque, out_values: *mut *mut *mut std::ffi::c_char, out_lengths: *mut *mut usize, out_count: *mut usize) -> i32;
    }
}

use std::ptr::NonNull;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CAbiError {
    pub operation: &'static str,
    pub status: i32,
}

pub struct Tensor {
    raw: NonNull<ffi::AtenTensorOpaque>,
}

pub struct TorchScriptModel {
    raw: NonNull<ffi::TerlanTorchScriptModelOpaque>,
}

pub struct TensorPair {
    raw: NonNull<ffi::TerlanTorchScriptTensorPairOpaque>,
}

struct DispatcherInputGuard {
    raw: Option<NonNull<ffi::AtenTensorOpaque>>,
}

impl DispatcherInputGuard {
    fn new(raw: *mut ffi::AtenTensorOpaque) -> Option<Self> {
        NonNull::new(raw).map(|raw| Self { raw: Some(raw) })
    }

    fn into_stable_ivalue(mut self) -> u64 {
        self.raw.take().expect("dispatcher input guard is armed").as_ptr() as usize as u64
    }
}

impl Drop for DispatcherInputGuard {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: an armed guard exclusively owns a duplicated handle not yet transferred to a dispatcher stack.
            let status = unsafe { ffi::aoti_torch_delete_tensor_object(raw.as_ptr()) };
            debug_assert_eq!(status, 0);
        }
    }
}

struct DispatcherOptionalValueGuard {
    raw: Option<NonNull<u64>>,
    destructor: unsafe extern "C" fn(*mut u64) -> i32,
    success: i32,
}

impl DispatcherOptionalValueGuard {
    fn new(
        raw: *mut u64,
        destructor: unsafe extern "C" fn(*mut u64) -> i32,
        success: i32,
    ) -> Option<Self> {
        NonNull::new(raw).map(|raw| Self { raw: Some(raw), destructor, success })
    }

    fn write_stable_ivalue(&mut self, value: u64) {
        // SAFETY: the allocator returned exclusive storage for one StableIValue.
        unsafe { *self.raw.expect("dispatcher optional guard is armed").as_ptr() = value };
    }

    fn write_i64(&mut self, value: i64) {
        self.write_stable_ivalue(value as u64);
    }

    fn into_stable_ivalue(mut self) -> u64 {
        self.raw.take().expect("dispatcher optional guard is armed").as_ptr() as usize as u64
    }
}

impl Drop for DispatcherOptionalValueGuard {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: an armed guard exclusively owns optional backing storage not yet transferred to the dispatcher.
            let status = unsafe { (self.destructor)(raw.as_ptr()) };
            debug_assert_eq!(status, self.success);
        }
    }
}

struct DispatcherListGuard {
    raw: Option<NonNull<()>>,
    destructor: unsafe extern "C" fn(*mut ()) -> i32,
    success: i32,
}

impl DispatcherListGuard {
    fn new(
        raw: *mut (),
        destructor: unsafe extern "C" fn(*mut ()) -> i32,
        success: i32,
    ) -> Option<Self> {
        NonNull::new(raw).map(|raw| Self { raw: Some(raw), destructor, success })
    }

    fn as_ptr(&self) -> *mut () {
        self.raw.expect("dispatcher list guard is armed").as_ptr()
    }

    fn into_stable_ivalue(mut self) -> u64 {
        self.raw.take().expect("dispatcher list guard is armed").as_ptr() as usize as u64
    }
}

impl Drop for DispatcherListGuard {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: an armed guard exclusively owns a dispatcher list not yet transferred to the stack.
            let status = unsafe { (self.destructor)(raw.as_ptr()) };
            debug_assert_eq!(status, self.success);
        }
    }
}

struct DispatcherStringGuard {
    raw: Option<NonNull<()>>,
    destructor: unsafe extern "C" fn(*mut ()) -> i32,
    success: i32,
}

impl DispatcherStringGuard {
    fn new(
        raw: *mut (),
        destructor: unsafe extern "C" fn(*mut ()) -> i32,
        success: i32,
    ) -> Option<Self> {
        NonNull::new(raw).map(|raw| Self { raw: Some(raw), destructor, success })
    }

    fn into_stable_ivalue(mut self) -> u64 {
        self.raw.take().expect("dispatcher string guard is armed").as_ptr() as usize as u64
    }
}

impl Drop for DispatcherStringGuard {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: an armed guard exclusively owns a dispatcher string not yet transferred to the stack.
            let status = unsafe { (self.destructor)(raw.as_ptr()) };
            debug_assert_eq!(status, self.success);
        }
    }
}

impl Tensor {
    pub fn from_int(value: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_scalar_to_tensor_int64(value as i64, &mut raw) };
        check_status("pytorch.tensor.from_int", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.from_int", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn from_float(value: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_scalar_to_tensor_float64(value as f64, &mut raw) };
        check_status("pytorch.tensor.from_float", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.from_float", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn from_bool(value: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_scalar_to_tensor_bool(value as bool, &mut raw) };
        check_status("pytorch.tensor.from_bool", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.from_bool", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn from_floats(values: &[f64], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.from_floats", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.from_floats", status: -2 })?;
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_tensor_from_f64(values.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("pytorch.tensor.from_floats", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.from_floats", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn from_ints(values: &[i64], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.from_ints", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.from_ints", status: -2 })?;
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_tensor_from_i64(values.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("pytorch.tensor.from_ints", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.from_ints", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn full_float(shape: &[i64], value: f64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.full_float", status: -2 })?;
        let mut fixed_dtype: i32 = 7;
        let mut fixed_device: i32 = 0;
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_aten_full(shape.as_ptr(), shape_length, value as f64, &mut fixed_dtype, std::ptr::null_mut(), &mut fixed_device, -1i32, std::ptr::null_mut(), &mut raw) };
        check_status("pytorch.tensor.full_float", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.full_float", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn item(&self) -> Result<i64, CAbiError> {
        let mut out_item: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_item_int64(self.raw.as_ptr(), &mut out_item) };
        check_status("pytorch.tensor.item", status as i32, 0)?;
        Ok(out_item as i64)
    }

    pub fn item_float(&self) -> Result<f64, CAbiError> {
        let mut out_item: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_item_float64(self.raw.as_ptr(), &mut out_item) };
        check_status("pytorch.tensor.item_float", status as i32, 0)?;
        Ok(out_item as f64)
    }

    pub fn item_bool(&self) -> Result<bool, CAbiError> {
        let mut out_item: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_item_bool(self.raw.as_ptr(), &mut out_item) };
        check_status("pytorch.tensor.item_bool", status as i32, 0)?;
        Ok(out_item)
    }

    pub fn dim(&self) -> Result<i64, CAbiError> {
        let mut out_dim: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_get_dim(self.raw.as_ptr(), &mut out_dim) };
        check_status("pytorch.tensor.dim", status as i32, 0)?;
        Ok(out_dim as i64)
    }

    pub fn numel(&self) -> Result<i64, CAbiError> {
        let mut out_numel: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_get_numel(self.raw.as_ptr(), &mut out_numel) };
        check_status("pytorch.tensor.numel", status as i32, 0)?;
        Ok(out_numel as i64)
    }

    pub fn shape(&self) -> Result<Vec<i64>, CAbiError> {
        let sizes_length = self.dim()?;
        let mut out_sizes: *mut i64 = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_get_sizes(self.raw.as_ptr(), &mut out_sizes) };
        check_status("pytorch.tensor.shape", status as i32, 0)?;
        let length = usize::try_from(sizes_length).map_err(|_| CAbiError { operation: "pytorch.tensor.shape", status: -2 })?;
        let values = if length == 0 {
            Vec::new()
        } else {
            let pointer = NonNull::new(out_sizes).ok_or(CAbiError { operation: "pytorch.tensor.shape", status: -1 })?;
            // SAFETY: metadata ties this borrowed array to `self`; it is copied before the borrow ends.
            unsafe { std::slice::from_raw_parts(pointer.as_ptr(), length).to_vec() }
        };
        Ok(values)
    }

    pub fn dtype(&self) -> Result<i64, CAbiError> {
        let mut out_dtype: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_get_dtype(self.raw.as_ptr(), &mut out_dtype) };
        check_status("pytorch.tensor.dtype", status as i32, 0)?;
        Ok(out_dtype as i64)
    }

    pub fn device_type(&self) -> Result<i64, CAbiError> {
        let mut out_device_type: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_get_device_type(self.raw.as_ptr(), &mut out_device_type) };
        check_status("pytorch.tensor.device_type", status as i32, 0)?;
        Ok(out_device_type as i64)
    }

    pub fn device_index(&self) -> Result<i64, CAbiError> {
        let mut out_device_index: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_get_device_index(self.raw.as_ptr(), &mut out_device_index) };
        check_status("pytorch.tensor.device_index", status as i32, 0)?;
        Ok(out_device_index as i64)
    }

    pub fn clone(&self) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.clone.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.clone", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue(), 0u64];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::clone\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.clone", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.clone", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn unsqueeze(&self, dimension: i64) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.unsqueeze.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.unsqueeze", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue(), dimension as u64];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::unsqueeze\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.unsqueeze", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.unsqueeze", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn matmul(&self, right: &Self) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_left: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_left) };
        check_status("pytorch.tensor.matmul.duplicate_handle.left", status as i32, 0)?;
        let dispatcher_input_left = DispatcherInputGuard::new(dispatcher_raw_left).ok_or(CAbiError { operation: "pytorch.tensor.matmul", status: -1 })?;
        let mut dispatcher_raw_right: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(right.raw.as_ptr(), &mut dispatcher_raw_right) };
        check_status("pytorch.tensor.matmul.duplicate_handle.right", status as i32, 0)?;
        let dispatcher_input_right = DispatcherInputGuard::new(dispatcher_raw_right).ok_or(CAbiError { operation: "pytorch.tensor.matmul", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_left.into_stable_ivalue(), dispatcher_input_right.into_stable_ivalue()];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::matmul\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.matmul", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.matmul", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn transpose(&self, dimension0: i64, dimension1: i64) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.transpose.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.transpose", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue(), dimension0 as u64, dimension1 as u64];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::transpose\0".as_ptr().cast(), "int\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.transpose", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.transpose", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn multiply(&self, right: &Self) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_left: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_left) };
        check_status("pytorch.tensor.multiply.duplicate_handle.left", status as i32, 0)?;
        let dispatcher_input_left = DispatcherInputGuard::new(dispatcher_raw_left).ok_or(CAbiError { operation: "pytorch.tensor.multiply", status: -1 })?;
        let mut dispatcher_raw_right: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(right.raw.as_ptr(), &mut dispatcher_raw_right) };
        check_status("pytorch.tensor.multiply.duplicate_handle.right", status as i32, 0)?;
        let dispatcher_input_right = DispatcherInputGuard::new(dispatcher_raw_right).ok_or(CAbiError { operation: "pytorch.tensor.multiply", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_left.into_stable_ivalue(), dispatcher_input_right.into_stable_ivalue()];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::mul\0".as_ptr().cast(), "Tensor\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.multiply", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.multiply", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn add(&self, right: &Self, alpha: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_cpu_add_Tensor(self.raw.as_ptr(), right.raw.as_ptr(), alpha as f64, &mut raw) };
        check_status("pytorch.tensor.add", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.add", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn subtract(&self, right: &Self, alpha: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_aten_subtract_Tensor(self.raw.as_ptr(), right.raw.as_ptr(), alpha as f64, &mut raw) };
        check_status("pytorch.tensor.subtract", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.subtract", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn relu(&self) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.relu.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.relu", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue()];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::relu\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.relu", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.relu", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn gelu(&self) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.gelu.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.gelu", status: -1 })?;
        let dispatcher_string_1_bytes: &[u8] = "none".as_bytes();
        let mut dispatcher_string_1_raw: *mut () = std::ptr::null_mut();
        // SAFETY: the reviewed allocator copies the fixed metadata bytes into one exclusive dispatcher string.
        let status = unsafe { ffi::torch_new_string_handle(dispatcher_string_1_bytes.as_ptr().cast(), dispatcher_string_1_bytes.len(), &mut dispatcher_string_1_raw) };
        check_status("pytorch.tensor.gelu.string.1.allocate", status as i32, 0)?;
        let dispatcher_string_1 = DispatcherStringGuard::new(dispatcher_string_1_raw, ffi::torch_delete_string, 0).ok_or(CAbiError { operation: "pytorch.tensor.gelu", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue(), dispatcher_string_1.into_stable_ivalue()];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::gelu\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.gelu", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.gelu", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn softmax(&self, dimension: i64) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.softmax.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.softmax", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue(), dimension as u64, 0u64];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::softmax\0".as_ptr().cast(), "int\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.softmax", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.softmax", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn argmax(&self, dimension: i64, keep_dimension: bool) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.argmax.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.argmax", status: -1 })?;
        let mut dispatcher_optional_dimension_raw: *mut u64 = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns exclusive storage for one StableIValue.
        let status = unsafe { ffi::torch_new_stable_ivalue(&mut dispatcher_optional_dimension_raw) };
        check_status("pytorch.tensor.argmax.optional.dimension.allocate", status as i32, 0)?;
        let mut dispatcher_optional_dimension = DispatcherOptionalValueGuard::new(dispatcher_optional_dimension_raw, ffi::torch_delete_stable_ivalue, 0).ok_or(CAbiError { operation: "pytorch.tensor.argmax", status: -1 })?;
        dispatcher_optional_dimension.write_i64(dimension);
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue(), dispatcher_optional_dimension.into_stable_ivalue(), u64::from(keep_dimension)];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::argmax\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.argmax", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.argmax", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn embedding(&self, indices: &Self, padding_index: i64, scale_gradient_by_frequency: bool, sparse: bool) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_weight: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_weight) };
        check_status("pytorch.tensor.embedding.duplicate_handle.weight", status as i32, 0)?;
        let dispatcher_input_weight = DispatcherInputGuard::new(dispatcher_raw_weight).ok_or(CAbiError { operation: "pytorch.tensor.embedding", status: -1 })?;
        let mut dispatcher_raw_indices: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(indices.raw.as_ptr(), &mut dispatcher_raw_indices) };
        check_status("pytorch.tensor.embedding.duplicate_handle.indices", status as i32, 0)?;
        let dispatcher_input_indices = DispatcherInputGuard::new(dispatcher_raw_indices).ok_or(CAbiError { operation: "pytorch.tensor.embedding", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_weight.into_stable_ivalue(), dispatcher_input_indices.into_stable_ivalue(), padding_index as u64, u64::from(scale_gradient_by_frequency), u64::from(sparse)];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::embedding\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.embedding", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.embedding", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn batch_norm(&self, weight: &Self, bias: &Self, running_mean: &Self, running_variance: &Self, training: bool, momentum: f64, epsilon: f64, cudnn_enabled: bool) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_input: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_input) };
        check_status("pytorch.tensor.batch_norm.duplicate_handle.input", status as i32, 0)?;
        let dispatcher_input_input = DispatcherInputGuard::new(dispatcher_raw_input).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_raw_weight: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(weight.raw.as_ptr(), &mut dispatcher_raw_weight) };
        check_status("pytorch.tensor.batch_norm.duplicate_handle.weight", status as i32, 0)?;
        let dispatcher_input_weight = DispatcherInputGuard::new(dispatcher_raw_weight).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_raw_bias: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(bias.raw.as_ptr(), &mut dispatcher_raw_bias) };
        check_status("pytorch.tensor.batch_norm.duplicate_handle.bias", status as i32, 0)?;
        let dispatcher_input_bias = DispatcherInputGuard::new(dispatcher_raw_bias).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_raw_running_mean: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(running_mean.raw.as_ptr(), &mut dispatcher_raw_running_mean) };
        check_status("pytorch.tensor.batch_norm.duplicate_handle.running_mean", status as i32, 0)?;
        let dispatcher_input_running_mean = DispatcherInputGuard::new(dispatcher_raw_running_mean).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_raw_running_variance: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(running_variance.raw.as_ptr(), &mut dispatcher_raw_running_variance) };
        check_status("pytorch.tensor.batch_norm.duplicate_handle.running_variance", status as i32, 0)?;
        let dispatcher_input_running_variance = DispatcherInputGuard::new(dispatcher_raw_running_variance).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_optional_weight_raw: *mut u64 = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns exclusive storage for one StableIValue.
        let status = unsafe { ffi::torch_new_stable_ivalue(&mut dispatcher_optional_weight_raw) };
        check_status("pytorch.tensor.batch_norm.optional.weight.allocate", status as i32, 0)?;
        let mut dispatcher_optional_weight = DispatcherOptionalValueGuard::new(dispatcher_optional_weight_raw, ffi::torch_delete_stable_ivalue, 0).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_optional_bias_raw: *mut u64 = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns exclusive storage for one StableIValue.
        let status = unsafe { ffi::torch_new_stable_ivalue(&mut dispatcher_optional_bias_raw) };
        check_status("pytorch.tensor.batch_norm.optional.bias.allocate", status as i32, 0)?;
        let mut dispatcher_optional_bias = DispatcherOptionalValueGuard::new(dispatcher_optional_bias_raw, ffi::torch_delete_stable_ivalue, 0).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_optional_running_mean_raw: *mut u64 = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns exclusive storage for one StableIValue.
        let status = unsafe { ffi::torch_new_stable_ivalue(&mut dispatcher_optional_running_mean_raw) };
        check_status("pytorch.tensor.batch_norm.optional.running_mean.allocate", status as i32, 0)?;
        let mut dispatcher_optional_running_mean = DispatcherOptionalValueGuard::new(dispatcher_optional_running_mean_raw, ffi::torch_delete_stable_ivalue, 0).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        let mut dispatcher_optional_running_variance_raw: *mut u64 = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns exclusive storage for one StableIValue.
        let status = unsafe { ffi::torch_new_stable_ivalue(&mut dispatcher_optional_running_variance_raw) };
        check_status("pytorch.tensor.batch_norm.optional.running_variance.allocate", status as i32, 0)?;
        let mut dispatcher_optional_running_variance = DispatcherOptionalValueGuard::new(dispatcher_optional_running_variance_raw, ffi::torch_delete_stable_ivalue, 0).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        dispatcher_optional_weight.write_stable_ivalue(dispatcher_input_weight.into_stable_ivalue());
        dispatcher_optional_bias.write_stable_ivalue(dispatcher_input_bias.into_stable_ivalue());
        dispatcher_optional_running_mean.write_stable_ivalue(dispatcher_input_running_mean.into_stable_ivalue());
        dispatcher_optional_running_variance.write_stable_ivalue(dispatcher_input_running_variance.into_stable_ivalue());
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_input.into_stable_ivalue(), dispatcher_optional_weight.into_stable_ivalue(), dispatcher_optional_bias.into_stable_ivalue(), dispatcher_optional_running_mean.into_stable_ivalue(), dispatcher_optional_running_variance.into_stable_ivalue(), u64::from(training), momentum.to_bits(), epsilon.to_bits(), u64::from(cudnn_enabled)];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::batch_norm\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.batch_norm", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.batch_norm", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn layer_norm(&self, normalized_shape: &[i64], weight: &Self, bias: &Self, epsilon: f64, cudnn_enabled: bool) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_input: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_input) };
        check_status("pytorch.tensor.layer_norm.duplicate_handle.input", status as i32, 0)?;
        let dispatcher_input_input = DispatcherInputGuard::new(dispatcher_raw_input).ok_or(CAbiError { operation: "pytorch.tensor.layer_norm", status: -1 })?;
        let mut dispatcher_raw_weight: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(weight.raw.as_ptr(), &mut dispatcher_raw_weight) };
        check_status("pytorch.tensor.layer_norm.duplicate_handle.weight", status as i32, 0)?;
        let dispatcher_input_weight = DispatcherInputGuard::new(dispatcher_raw_weight).ok_or(CAbiError { operation: "pytorch.tensor.layer_norm", status: -1 })?;
        let mut dispatcher_raw_bias: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(bias.raw.as_ptr(), &mut dispatcher_raw_bias) };
        check_status("pytorch.tensor.layer_norm.duplicate_handle.bias", status as i32, 0)?;
        let dispatcher_input_bias = DispatcherInputGuard::new(dispatcher_raw_bias).ok_or(CAbiError { operation: "pytorch.tensor.layer_norm", status: -1 })?;
        let mut dispatcher_optional_weight_raw: *mut u64 = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns exclusive storage for one StableIValue.
        let status = unsafe { ffi::torch_new_stable_ivalue(&mut dispatcher_optional_weight_raw) };
        check_status("pytorch.tensor.layer_norm.optional.weight.allocate", status as i32, 0)?;
        let mut dispatcher_optional_weight = DispatcherOptionalValueGuard::new(dispatcher_optional_weight_raw, ffi::torch_delete_stable_ivalue, 0).ok_or(CAbiError { operation: "pytorch.tensor.layer_norm", status: -1 })?;
        let mut dispatcher_optional_bias_raw: *mut u64 = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns exclusive storage for one StableIValue.
        let status = unsafe { ffi::torch_new_stable_ivalue(&mut dispatcher_optional_bias_raw) };
        check_status("pytorch.tensor.layer_norm.optional.bias.allocate", status as i32, 0)?;
        let mut dispatcher_optional_bias = DispatcherOptionalValueGuard::new(dispatcher_optional_bias_raw, ffi::torch_delete_stable_ivalue, 0).ok_or(CAbiError { operation: "pytorch.tensor.layer_norm", status: -1 })?;
        let mut dispatcher_list_normalized_shape_raw: *mut () = std::ptr::null_mut();
        // SAFETY: the reviewed allocator returns one exclusive dispatcher list.
        let status = unsafe { ffi::torch_new_list_reserve_size(normalized_shape.len(), &mut dispatcher_list_normalized_shape_raw) };
        check_status("pytorch.tensor.layer_norm.list.normalized_shape.allocate", status as i32, 0)?;
        let dispatcher_list_normalized_shape = DispatcherListGuard::new(dispatcher_list_normalized_shape_raw, ffi::torch_delete_list, 0).ok_or(CAbiError { operation: "pytorch.tensor.layer_norm", status: -1 })?;
        for element in normalized_shape {
            // SAFETY: the armed list guard owns the destination and integer StableIValues are copied by value.
            let status = unsafe { ffi::torch_list_push_back(dispatcher_list_normalized_shape.as_ptr(), *element as u64) };
            check_status("pytorch.tensor.layer_norm.list.normalized_shape.push", status as i32, 0)?;
        }
        dispatcher_optional_weight.write_stable_ivalue(dispatcher_input_weight.into_stable_ivalue());
        dispatcher_optional_bias.write_stable_ivalue(dispatcher_input_bias.into_stable_ivalue());
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_input.into_stable_ivalue(), dispatcher_list_normalized_shape.into_stable_ivalue(), dispatcher_optional_weight.into_stable_ivalue(), dispatcher_optional_bias.into_stable_ivalue(), epsilon.to_bits(), u64::from(cudnn_enabled)];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::layer_norm\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.layer_norm", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.layer_norm", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn mse_loss(&self, target: &Self, reduction: i64) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_predicted: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_predicted) };
        check_status("pytorch.tensor.mse_loss.duplicate_handle.predicted", status as i32, 0)?;
        let dispatcher_input_predicted = DispatcherInputGuard::new(dispatcher_raw_predicted).ok_or(CAbiError { operation: "pytorch.tensor.mse_loss", status: -1 })?;
        let mut dispatcher_raw_target: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(target.raw.as_ptr(), &mut dispatcher_raw_target) };
        check_status("pytorch.tensor.mse_loss.duplicate_handle.target", status as i32, 0)?;
        let dispatcher_input_target = DispatcherInputGuard::new(dispatcher_raw_target).ok_or(CAbiError { operation: "pytorch.tensor.mse_loss", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_predicted.into_stable_ivalue(), dispatcher_input_target.into_stable_ivalue(), reduction as u64];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::mse_loss\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.mse_loss", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.mse_loss", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn mse_loss_gradient(&self, predicted: &Self, target: &Self, reduction: i64) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_grad_output: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_grad_output) };
        check_status("pytorch.tensor.mse_loss_gradient.duplicate_handle.grad_output", status as i32, 0)?;
        let dispatcher_input_grad_output = DispatcherInputGuard::new(dispatcher_raw_grad_output).ok_or(CAbiError { operation: "pytorch.tensor.mse_loss_gradient", status: -1 })?;
        let mut dispatcher_raw_predicted: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(predicted.raw.as_ptr(), &mut dispatcher_raw_predicted) };
        check_status("pytorch.tensor.mse_loss_gradient.duplicate_handle.predicted", status as i32, 0)?;
        let dispatcher_input_predicted = DispatcherInputGuard::new(dispatcher_raw_predicted).ok_or(CAbiError { operation: "pytorch.tensor.mse_loss_gradient", status: -1 })?;
        let mut dispatcher_raw_target: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(target.raw.as_ptr(), &mut dispatcher_raw_target) };
        check_status("pytorch.tensor.mse_loss_gradient.duplicate_handle.target", status as i32, 0)?;
        let dispatcher_input_target = DispatcherInputGuard::new(dispatcher_raw_target).ok_or(CAbiError { operation: "pytorch.tensor.mse_loss_gradient", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_grad_output.into_stable_ivalue(), dispatcher_input_predicted.into_stable_ivalue(), dispatcher_input_target.into_stable_ivalue(), reduction as u64];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::mse_loss_backward\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.mse_loss_gradient", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.mse_loss_gradient", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn cross_entropy_loss(&self, target: &Self, reduction: i64, ignore_index: i64, label_smoothing: f64) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_logits: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_logits) };
        check_status("pytorch.tensor.cross_entropy_loss.duplicate_handle.logits", status as i32, 0)?;
        let dispatcher_input_logits = DispatcherInputGuard::new(dispatcher_raw_logits).ok_or(CAbiError { operation: "pytorch.tensor.cross_entropy_loss", status: -1 })?;
        let mut dispatcher_raw_target: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(target.raw.as_ptr(), &mut dispatcher_raw_target) };
        check_status("pytorch.tensor.cross_entropy_loss.duplicate_handle.target", status as i32, 0)?;
        let dispatcher_input_target = DispatcherInputGuard::new(dispatcher_raw_target).ok_or(CAbiError { operation: "pytorch.tensor.cross_entropy_loss", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_logits.into_stable_ivalue(), dispatcher_input_target.into_stable_ivalue(), 0u64, reduction as u64, ignore_index as u64, label_smoothing.to_bits()];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::cross_entropy_loss\0".as_ptr().cast(), "\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.cross_entropy_loss", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.cross_entropy_loss", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn narrow(&self, dimension: i64, start: i64, length: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_aten_narrow(self.raw.as_ptr(), dimension as i64, start as i64, length as i64, &mut raw) };
        check_status("pytorch.tensor.narrow", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.narrow", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn flatten(&self, start_dimension: i64, end_dimension: i64) -> Result<Self, CAbiError> {
        let mut dispatcher_raw_tensor: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: the source handle is borrowed and the returned handle is an independent owner.
        let status = unsafe { ffi::aoti_torch_new_tensor_handle(self.raw.as_ptr(), &mut dispatcher_raw_tensor) };
        check_status("pytorch.tensor.flatten.duplicate_handle.tensor", status as i32, 0)?;
        let dispatcher_input_tensor = DispatcherInputGuard::new(dispatcher_raw_tensor).ok_or(CAbiError { operation: "pytorch.tensor.flatten", status: -1 })?;
        // StableIValue takes ownership of copied handle inputs; the selected output slot transfers ownership back.
        let mut stack = [dispatcher_input_tensor.into_stable_ivalue(), start_dimension as u64, end_dimension as u64];
        // SAFETY: metadata fixes the operator schema, stack layout, ABI version, and ownership transfer.
        let status = unsafe { ffi::torch_call_dispatcher("aten::flatten\0".as_ptr().cast(), "using_ints\0".as_ptr().cast(), stack.as_mut_ptr(), 0x020d000000000000u64) };
        check_status("pytorch.tensor.flatten", status as i32, 0)?;
        let raw = stack[0] as usize as *mut ffi::AtenTensorOpaque;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.flatten", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn amax(&self, dimensions: &[i64], keep_dimension: bool) -> Result<Self, CAbiError> {
        let dimensions_length = i64::try_from(dimensions.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.amax", status: -2 })?;
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_aten_amax(self.raw.as_ptr(), dimensions.as_ptr(), dimensions_length, keep_dimension as i32, &mut raw) };
        check_status("pytorch.tensor.amax", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.amax", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn conv2d(&self, weight: &Self, bias: &Self, stride: &[i64], padding: &[i64], dilation: &[i64], groups: i64) -> Result<Self, CAbiError> {
        let stride_length = i64::try_from(stride.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.conv2d", status: -2 })?;
        let padding_length = i64::try_from(padding.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.conv2d", status: -2 })?;
        let dilation_length = i64::try_from(dilation.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.conv2d", status: -2 })?;
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_conv2d(self.raw.as_ptr(), weight.raw.as_ptr(), bias.raw.as_ptr(), stride.as_ptr(), stride_length, padding.as_ptr(), padding_length, dilation.as_ptr(), dilation_length, groups as i64, &mut raw) };
        check_status("pytorch.tensor.conv2d", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.conv2d", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn adaptive_avg_pool2d(&self, output_size: &[i64]) -> Result<Self, CAbiError> {
        let output_size_length = i64::try_from(output_size.len()).map_err(|_| CAbiError { operation: "pytorch.tensor.adaptive_avg_pool2d", status: -2 })?;
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::aoti_torch_cpu__adaptive_avg_pool2d(self.raw.as_ptr(), output_size.as_ptr(), output_size_length, &mut raw) };
        check_status("pytorch.tensor.adaptive_avg_pool2d", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.adaptive_avg_pool2d", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn run_torchscript(&self, path: &str) -> Result<Self, CAbiError> {
        let path_c = std::ffi::CString::new(path.as_bytes()).map_err(|_| CAbiError { operation: "pytorch.tensor.run_torchscript", status: -3 })?;
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_run_torchscript(self.raw.as_ptr(), path_c.as_ptr(), &mut raw) };
        check_status("pytorch.tensor.run_torchscript", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.tensor.run_torchscript", status: -1 })?;
        Ok(Self { raw })
    }

}

impl Drop for Tensor {
    fn drop(&mut self) {
        // SAFETY: this adapter is the sole owner and invokes the destructor once.
        let status = unsafe { ffi::aoti_torch_delete_tensor_object(self.raw.as_ptr()) };
        debug_assert_eq!(status, 0);
    }
}

impl TorchScriptModel {
    pub fn load_torchscript(path: &str) -> Result<Self, CAbiError> {
        let path_c = std::ffi::CString::new(path.as_bytes()).map_err(|_| CAbiError { operation: "pytorch.torchscript.load", status: -3 })?;
        let mut raw: *mut ffi::TerlanTorchScriptModelOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_load_torchscript(path_c.as_ptr(), &mut raw) };
        check_status("pytorch.torchscript.load", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.torchscript.load", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn torchscript_forward(&self, input: &Tensor) -> Result<Tensor, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward(self.raw.as_ptr(), input.raw.as_ptr(), &mut raw) };
        check_status("pytorch.torchscript.forward", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.torchscript.forward", status: -1 })?;
        Ok(Tensor { raw })
    }

    pub fn torchscript_forward2(&self, first: &Tensor, second: &Tensor) -> Result<Tensor, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward2(self.raw.as_ptr(), first.raw.as_ptr(), second.raw.as_ptr(), &mut raw) };
        check_status("pytorch.torchscript.forward2", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.torchscript.forward2", status: -1 })?;
        Ok(Tensor { raw })
    }

    pub fn torchscript_forward_pair(&self, input: &Tensor) -> Result<TensorPair, CAbiError> {
        let mut raw: *mut ffi::TerlanTorchScriptTensorPairOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_pair(self.raw.as_ptr(), input.raw.as_ptr(), &mut raw) };
        check_status("pytorch.torchscript.forward_pair", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.torchscript.forward_pair", status: -1 })?;
        Ok(TensorPair { raw })
    }

    pub fn torchscript_forward_float(&self, input: &Tensor) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_float(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_value) };
        check_status("pytorch.torchscript.forward_float", status as i32, 0)?;
        Ok(out_out_value as f64)
    }

    pub fn torchscript_forward_int(&self, input: &Tensor) -> Result<i64, CAbiError> {
        let mut out_out_value: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_int(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_value) };
        check_status("pytorch.torchscript.forward_int", status as i32, 0)?;
        Ok(out_out_value as i64)
    }

    pub fn torchscript_forward_bool(&self, input: &Tensor) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_bool(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_value) };
        check_status("pytorch.torchscript.forward_bool", status as i32, 0)?;
        Ok(out_out_value)
    }

    pub fn torchscript_forward_string(&self, input: &Tensor) -> Result<String, CAbiError> {
        let mut out_out_value: *mut std::ffi::c_char = std::ptr::null_mut();
        let mut out_out_value_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_string(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_value, &mut out_out_value_length) };
        check_status("pytorch.torchscript.forward_string", status as i32, 0)?;
        let bytes = if out_out_value_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_value).map(|pointer| {
                // SAFETY: the producer returned `out_out_value_length` initialized bytes; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr().cast::<u8>(), out_out_value_length).to_vec() }
            }).ok_or(CAbiError { operation: "pytorch.torchscript.forward_string", status: -1 })
        };
        // SAFETY: ownership metadata transfers this pointer to the named destructor exactly once.
        unsafe { ffi::terlan_pytorch_delete_owned_string(out_out_value) };
        let bytes = bytes?;
        let value = String::from_utf8(bytes).map_err(|_| CAbiError { operation: "pytorch.torchscript.forward_string", status: -3 })?;
        Ok(value)
    }

    pub fn torchscript_forward_int_list(&self, input: &Tensor) -> Result<Vec<i64>, CAbiError> {
        let mut out_out_values: *mut i64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_int_list(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("pytorch.torchscript.forward_int_list", status as i32, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "pytorch.torchscript.forward_int_list", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_pytorch_delete_owned_int_array(out_out_values) };
        Ok(values?)
    }

    pub fn torchscript_forward_float_list(&self, input: &Tensor) -> Result<Vec<f64>, CAbiError> {
        let mut out_out_values: *mut f64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_float_list(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("pytorch.torchscript.forward_float_list", status as i32, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "pytorch.torchscript.forward_float_list", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_pytorch_delete_owned_float_array(out_out_values) };
        Ok(values?)
    }

    pub fn torchscript_forward_bool_list(&self, input: &Tensor) -> Result<Vec<bool>, CAbiError> {
        let mut out_out_values: *mut u8 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_bool_list(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("pytorch.torchscript.forward_bool_list", status as i32, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "pytorch.torchscript.forward_bool_list", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_pytorch_delete_owned_bool_array(out_out_values) };
        let values = values?;
        values.into_iter().map(|value| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CAbiError { operation: "pytorch.torchscript.forward_bool_list", status: -4 }),
        }).collect()
    }

    pub fn torchscript_forward_string_list(&self, input: &Tensor) -> Result<Vec<String>, CAbiError> {
        let mut out_out_values: *mut *mut std::ffi::c_char = std::ptr::null_mut();
        let mut out_out_values_lengths: *mut usize = std::ptr::null_mut();
        let mut out_out_values_count: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_torchscript_forward_string_list(self.raw.as_ptr(), input.raw.as_ptr(), &mut out_out_values, &mut out_out_values_lengths, &mut out_out_values_count) };
        check_status("pytorch.torchscript.forward_string_list", status as i32, 0)?;
        let bytes: Result<Vec<Vec<u8>>, CAbiError> = if out_out_values_count == 0 {
            Ok(Vec::new())
        } else {
            match (NonNull::new(out_out_values), NonNull::new(out_out_values_lengths)) {
                (Some(values_pointer), Some(lengths_pointer)) => {
                    // SAFETY: the producer returned `out_out_values_count` initialized pointers and lengths; every string is copied before destruction.
                    let value_pointers = unsafe { std::slice::from_raw_parts(values_pointer.as_ptr(), out_out_values_count) };
                    let value_lengths = unsafe { std::slice::from_raw_parts(lengths_pointer.as_ptr(), out_out_values_count) };
                    value_pointers.iter().zip(value_lengths).map(|(&value_pointer, &value_length)| {
                        if value_length == 0 {
                            Ok(Vec::new())
                        } else {
                            let value_pointer = NonNull::new(value_pointer).ok_or(CAbiError { operation: "pytorch.torchscript.forward_string_list", status: -1 })?;
                            // SAFETY: the matching length describes this initialized string allocation.
                            Ok(unsafe { std::slice::from_raw_parts(value_pointer.as_ptr().cast::<u8>(), value_length).to_vec() })
                        }
                    }).collect()
                }
                _ => Err(CAbiError { operation: "pytorch.torchscript.forward_string_list", status: -1 }),
            }
        };
        // SAFETY: ownership metadata transfers both arrays and their string elements to the named destructor exactly once.
        unsafe { ffi::terlan_pytorch_delete_owned_string_array(out_out_values, out_out_values_lengths, out_out_values_count) };
        let bytes = bytes?;
        bytes.into_iter().map(|value| String::from_utf8(value).map_err(|_| CAbiError { operation: "pytorch.torchscript.forward_string_list", status: -3 })).collect()
    }

}

impl Drop for TorchScriptModel {
    fn drop(&mut self) {
        // SAFETY: this adapter is the sole owner and invokes the destructor once.
        let status = unsafe { ffi::terlan_pytorch_delete_torchscript_model(self.raw.as_ptr()) };
        debug_assert_eq!(status, 0);
    }
}

impl TensorPair {
    pub fn tensor_pair_first(&self) -> Result<Tensor, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_tensor_pair_first(self.raw.as_ptr(), &mut raw) };
        check_status("pytorch.torchscript.tensor_pair_first", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.torchscript.tensor_pair_first", status: -1 })?;
        Ok(Tensor { raw })
    }

    pub fn tensor_pair_second(&self) -> Result<Tensor, CAbiError> {
        let mut raw: *mut ffi::AtenTensorOpaque = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_pytorch_tensor_pair_second(self.raw.as_ptr(), &mut raw) };
        check_status("pytorch.torchscript.tensor_pair_second", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "pytorch.torchscript.tensor_pair_second", status: -1 })?;
        Ok(Tensor { raw })
    }

}

impl Drop for TensorPair {
    fn drop(&mut self) {
        // SAFETY: this adapter is the sole owner and invokes the destructor once.
        let status = unsafe { ffi::terlan_pytorch_delete_tensor_pair(self.raw.as_ptr()) };
        debug_assert_eq!(status, 0);
    }
}

pub fn abi_version() -> i64 {
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let value = unsafe { ffi::aoti_torch_abi_version() };
    value as i64
}

fn check_status(operation: &'static str, status: i32, success: i32) -> Result<(), CAbiError> {
    if status == success {
        Ok(())
    } else {
        Err(CAbiError { operation, status })
    }
}
