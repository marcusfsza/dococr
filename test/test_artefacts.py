import os
from doctr.models.artefacts import BarCodeDetector
from doctr.documents import DocumentFile


def test_qr_code_detector(mock_image_folder):
    detector = BarCodeDetector()
    for img in os.listdir(mock_image_folder):
        image = DocumentFile.from_images(os.path.join(mock_image_folder, img))[0]
        barcode = detector(image)
        assert len(barcode) == 0
