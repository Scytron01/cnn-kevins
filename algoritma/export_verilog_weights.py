from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from algoritma.model import LaundryShapeCNN

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "report/runs/laundry_shape_cnn/best_model.pt"
DEFAULT_DATA = ROOT / "dataset/processed/laundry64.npz"
DEFAULT_OUT = ROOT / "verilog/weights"
Q24 = 1 << 24
Q7_MAX = 127

def activation_scale(maximum: float, margin: float = 1.10) -> float:
    return max(float(maximum) * margin / Q7_MAX, 1e-8)

def quantize_weight(weight: torch.Tensor):
    values = weight.detach().cpu().numpy().astype(np.float64)
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    scale = maximum / Q7_MAX if maximum > 0.0 else 1.0 / Q7_MAX
    return np.rint(values / scale).clip(-127, 127).astype(np.int8), scale

def quantize_bias(bias: torch.Tensor, input_scale: float, weight_scale: float):
    values = bias.detach().cpu().numpy().astype(np.float64)
    quantized = np.rint(values / (input_scale * weight_scale))
    if np.any(quantized < -(2**31)) or np.any(quantized > (2**31 - 1)):
        raise OverflowError("Bias tidak muat pada signed int32. Periksa skala kuantisasi.")
    return quantized.astype(np.int64)

def q24_multiplier(real_multiplier: float, name: str) -> int:
    value = max(int(np.rint(real_multiplier * Q24)), 1)
    if value >= 2**31:
        raise OverflowError(f"Multiplier {name}={real_multiplier:.8g} tidak muat pada Q24 positif 31-bit.")
    return value

def write_signed_hex(path: Path, values: np.ndarray, bits: int):
    width = bits // 4
    mask = (1 << bits) - 1
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n" for v in np.asarray(values).reshape(-1)), encoding="ascii")

def write_unsigned_word(path: Path, value: int):
    path.write_text(f"{int(value) & 0xFFFFFFFF:08x}\n", encoding="ascii")

def calibrate(model: LaundryShapeCNN, data_path: Path, max_samples: int, batch_size: int):
    archive = np.load(data_path, allow_pickle=False)
    masks = archive["X_mask"][:max_samples].astype(np.float32)
    geometry = archive["X_geom"][:max_samples].astype(np.float32)
    if len(masks) == 0:
        raise ValueError("Dataset kalibrasi kosong.")
    if masks.shape[1:] != (1,64,64) or geometry.shape != (len(masks),4):
        raise ValueError("Shape kalibrasi harus X_mask=(N,1,64,64), X_geom=(N,4).")
    maxima = {"conv1":0.0,"conv2":0.0,"conv3":0.0,"gap":0.0,"geometry":0.0,"fc1":0.0}
    model.eval()
    with torch.no_grad():
        for start in range(0,len(masks),batch_size):
            x = torch.from_numpy(masks[start:start+batch_size])
            g = torch.from_numpy(geometry[start:start+batch_size])
            c1 = model.features[0](x)
            c2 = model.features[1](c1)
            c3 = model.features[2](c2)
            gap = model.gap(c3).flatten(1)
            joined = torch.cat((gap,g),dim=1)
            fc1 = F.relu(model.classifier[0](joined))
            maxima["conv1"] = max(maxima["conv1"],float(c1.max().item()))
            maxima["conv2"] = max(maxima["conv2"],float(c2.max().item()))
            maxima["conv3"] = max(maxima["conv3"],float(c3.max().item()))
            maxima["gap"] = max(maxima["gap"],float(gap.max().item()))
            maxima["geometry"] = max(maxima["geometry"],float(g.max().item()))
            maxima["fc1"] = max(maxima["fc1"],float(fc1.max().item()))
    scales = {
        "input":1.0,
        "conv1":activation_scale(maxima["conv1"]),
        "conv2":activation_scale(maxima["conv2"]),
        "conv3":activation_scale(maxima["conv3"]),
        "fc_input":activation_scale(max(maxima["gap"],maxima["geometry"])),
        "fc1":activation_scale(maxima["fc1"]),
    }
    return scales,maxima,len(masks)

def export(checkpoint: Path,data_path: Path,output_dir: Path,max_calibration_samples: int,batch_size: int):
    saved = torch.load(checkpoint,map_location="cpu",weights_only=False)
    class_names = saved.get("class_names",[])
    if len(class_names)!=5:
        raise ValueError("Checkpoint harus memiliki tepat lima class_names.")
    model = LaundryShapeCNN(5)
    model.load_state_dict(saved["model_state_dict"],strict=True)
    model.eval()
    scales,maxima,sample_count = calibrate(model,data_path,max_calibration_samples,batch_size)
    state = model.state_dict()
    layers = [
        ("conv1","features.0.net.0.weight","features.0.net.0.bias",scales["input"],scales["conv1"]),
        ("conv2","features.1.net.0.weight","features.1.net.0.bias",scales["conv1"],scales["conv2"]),
        ("conv3","features.2.net.0.weight","features.2.net.0.bias",scales["conv2"],scales["conv3"]),
        ("fc1","classifier.0.weight","classifier.0.bias",scales["fc_input"],scales["fc1"]),
        ("fc2","classifier.3.weight","classifier.3.bias",scales["fc1"],None),
    ]
    output_dir.mkdir(parents=True,exist_ok=True)
    metadata = {
        "source_checkpoint":str(checkpoint),"calibration_dataset":str(data_path),
        "calibration_samples":sample_count,"class_names":class_names,
        "activation_encoding":"Conv1 input is binary 0/1 without multiplier; later activations are nonnegative INT8 codes 0..127 with per-layer calibrated scales",
        "weight_encoding":"symmetric signed int8, per-layer scale max_abs(weight)/127",
        "accumulator":"signed int32","requant_multiplier":"unsigned Q24",
        "activation_maxima":maxima,"activation_scales_real_per_code":scales,"layers":{}
    }
    for name,wkey,bkey,input_scale,output_scale in layers:
        qw,ws = quantize_weight(state[wkey])
        qb = quantize_bias(state[bkey],input_scale,ws)
        write_signed_hex(output_dir/f"{name}_w.mem",qw,8)
        write_signed_hex(output_dir/f"{name}_b.mem",qb,32)
        metadata["layers"][name]={"weight_shape":list(state[wkey].shape),
            "input_scale":input_scale,"weight_scale":ws,"bias_scale":input_scale*ws}
        if output_scale is not None:
            mul = q24_multiplier(input_scale*ws/output_scale,name)
            write_unsigned_word(output_dir/f"{name}_mul.mem",mul)
            metadata["layers"][name]["output_scale"]=output_scale
            metadata["layers"][name]["requant_multiplier_q24"]=mul
        else:
            metadata["layers"][name]["output"]="raw logits; shared scale makes ArgMax comparison valid"
    gap_mul = q24_multiplier(scales["conv3"]/scales["fc_input"],"gap")
    geom_mul = q24_multiplier((1.0/Q7_MAX)/scales["fc_input"],"geometry")
    write_unsigned_word(output_dir/"gap_mul.mem",gap_mul)
    write_unsigned_word(output_dir/"geom_mul.mem",geom_mul)
    metadata["gap_requant_multiplier_q24"]=gap_mul
    metadata["geometry_requant_multiplier_q24"]=geom_mul
    metadata["notes"]=[
        "Convolutions use SAME padding and 2x2 max-pooling.",
        "Feature maps are stored channel-last in cnn_top.",
        "FC2 outputs are raw signed accumulators; common output scale makes ArgMax valid.",
        "Conv1 input is stored as binary 0/1; geometry_q7 bytes are round(normalized_feature*127), clamped to 0..127."
    ]
    (output_dir/"quantization.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    print("[EXPORT] Wrote ROM files to",output_dir)
    print("[EXPORT] Calibration samples:",sample_count)
    print("[EXPORT] Activation scales:",scales)
    return metadata

def main():
    parser=argparse.ArgumentParser(description="Export trained LaundryShapeCNN weights to Verilog ROM .mem files")
    parser.add_argument("--checkpoint",type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--max-calibration-samples",type=int,default=2048)
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.checkpoint}. Jalankan training dahulu.")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset kalibrasi tidak ditemukan: {args.data}. Jalankan prepare_dataset dahulu.")
    export(args.checkpoint,args.data,args.out_dir,args.max_calibration_samples,args.batch_size)

if __name__=="__main__":
    main()
