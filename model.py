from torch import argmax, load, device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50
from PIL import Image
from io import BytesIO

LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

device = "cuda" if is_available() else "cpu"

# Model definition
resnet_model = resnet50(pretrained=True)
n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(
    Linear(n_inputs, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 4),
    LogSigmoid()
)

resnet_model.load_state_dict(load('./models/bt_resnet50_model.pt', map_location=DEVICE(device)))
resnet_model.to(device)
resnet_model.eval()

# Image preprocessing
def preprocess_image(image_bytes):
    transform = Compose([Resize((512, 512)), ToTensor()])
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)

# Prediction
def get_prediction(image_bytes):
    tensor = preprocess_image(image_bytes)
    with DEVICE(device):
        y_hat = resnet_model(tensor.to(device))
    class_id = argmax(y_hat.data, dim=1)
    return str(int(class_id)), LABELS[int(class_id)]
