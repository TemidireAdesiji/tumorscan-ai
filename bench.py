import time, torch, onnxruntime as ort
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from torchvision.models import resnet50

model = resnet50(weights=None)
model.fc = Sequential(
    Linear(model.fc.in_features, 2048), SELU(), Dropout(0.4),
    Linear(2048, 2048), SELU(), Dropout(0.4),
    Linear(2048, 4), LogSigmoid(),
)
model.eval()
x = torch.randn(1, 3, 512, 512)
torch.onnx.export(model, x, "tumorscan.onnx",
                  input_names=["input"], output_names=["output"], opset_version=18)

def bench(fn, n=50):
    for _ in range(10): fn()
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000

xn = x.numpy()
with torch.no_grad():
    pt_cpu = bench(lambda: model(x))
    model.to("mps"); xm = x.to("mps")
    def run_mps():
        model(xm); torch.mps.synchronize()
    pt_gpu = bench(run_mps)

ort_cpu = bench(lambda: ort.InferenceSession("tumorscan.onnx",
    providers=["CPUExecutionProvider"]).run if False else None) if False else None
cpu_sess = ort.InferenceSession("tumorscan.onnx", providers=["CPUExecutionProvider"])
ort_cpu = bench(lambda: cpu_sess.run(None, {"input": xn}))
cml_sess = ort.InferenceSession("tumorscan.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
ort_cml = bench(lambda: cml_sess.run(None, {"input": xn}))

print(f"PyTorch CPU      {pt_cpu:.1f} ms")
print(f"PyTorch Apple GPU {pt_gpu:.1f} ms")
print(f"ONNX Runtime CPU {ort_cpu:.1f} ms")
print(f"ONNX + CoreML    {ort_cml:.1f} ms")
print(f"ONNX+CoreML vs PyTorch GPU: {100*(1-ort_cml/pt_gpu):.0f}% faster")
