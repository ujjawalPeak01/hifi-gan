import os
import base64
import torch
import requests
from models import Generator
from inference import initialize_helper, inference


class InferlessPythonModel:
    def initialize(self):
        initialize_helper("generator_v3")

    def dowload_wav_file(self, audio_url):
        file_name = audio_url.split("/")[-1]
        save_path = os.path.join("test_files", file_name)
        response = requests.get(audio_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"File downloaded successfully and saved to {save_path}")
        else:
            print(
                f"Failed to download the file. HTTP Status Code: {response.status_code}"
            )

    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        self.dowload_wav_file(audio_url)

        inference("generator_v3", "test_files", "generated_files")

        file_name = audio_url.split("/")[-1]
        file_path = "generated_files/" + file_name.split(".")[0] + "_generated.wav"

        with open(file_path, "rb") as f:
            audio_data = f.read()

            base64_audio = base64.b64encode(audio_data).decode("utf-8")

        return {"generated_base_64": base64_audio}

    def finalize():
        pass
