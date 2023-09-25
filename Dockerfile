FROM nvcr.io/nvidia/tritonserver:22.11-py3

RUN apt update && apt -y install libssl-dev tesseract-ocr libtesseract-dev ffmpeg

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install  --no-cache-dir  -r requirements.txt

COPY . .

CMD python3 inference.py --checkpoint_file generator_v3