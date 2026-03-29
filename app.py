from flask import Flask, request, render_template
import torch
from models.multimodal_model import MultimodalModel
from utils.text_utils import get_text_features
from utils.image_utils import get_image_features

app = Flask(__name__)

# Load model
model = MultimodalModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

labels = ["Negative", "Neutral", "Positive"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        image = request.files["image"]

        image_path = "temp.jpg"
        image.save(image_path)

        img_feat = get_image_features(image_path)
        text_feat = get_text_features(text)

        output = model(img_feat, text_feat)
        pred = torch.argmax(output, dim=1).item()

        return render_template("index.html", result=labels[pred])

    return render_template("index.html")

if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
