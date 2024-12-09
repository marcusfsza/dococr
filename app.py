from flask import Flask, request, jsonify
from PIL import Image
import io
from transformers import AutoProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# Carica il processor e il modello. Assicurati di avere una connessione a Internet la prima volta.
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR2_0", trust_remote_code=True)
model = VisionEncoderDecoderModel.from_pretrained("stepfun-ai/GOT-OCR2_0", trust_remote_code=True)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "Nessun file inviato"}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    inputs = processor(img, return_tensors="pt")
    generated_ids = model.generate(**inputs)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"text": text})

if __name__ == '__main__':
    # Avvia il server Flask
    # Accessibile su http://127.0.0.1:5000/ocr
    app.run(host="127.0.0.1", port=5000, debug=True)
