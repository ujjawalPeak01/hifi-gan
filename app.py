import os
import base64
import torch
import shutil
import requests
import soundfile as sf
from io import BytesIO
from models import Generator
from inference import initialize_helper, inference


class InferlessPythonModel:
    def initialize(self):
            volume_location = '/var/nfs-mount/new-vol'
            current_dir_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            self.model_weights_file_name = "generator_v3"
            self.config_file_name = "config.json"
            self.location = volume_location

            model_weights_path = os.path.join(volume_location, self.model_weights_file_name)
            config_file_path = os.path.join(volume_location, self.config_file_name)
            
            if not os.path.exists(model_weights_path) or not os.path.exists(config_file_path):
                src = os.path.join(current_dir_location, self.model_weights_file_name)
                src_config = os.path.join(current_dir_location, self.config_file_name)
                dst = volume_location
    
                shutil.copy(src, dst)
                shutil.copy(src_config, dst)
            
            initialize_helper(os.path.join(self.location, self.model_weights_file_name))

    def dowload_wav_file(self, audio_url):
        response = requests.get(audio_url)
        if response.status_code == 200:
            audio_data = BytesIO(response.content)
            print("File downloaded successfully and kept in memory.")
        else:
            print(f"Failed to download the file. HTTP Status Code: {response.status_code}")
        return audio_data

    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        audio_data = self.dowload_wav_file(audio_url)

        result, sr = inference(
            os.path.join(self.location, self.model_weights_file_name),
            audio_data,
        )

        buffer = BytesIO()
        sf.write(buffer, result, sr, format='WAV')
        buffer.seek(0)

        base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        print(base64_audio)
        return {"generated_base_64": base64_audio}

    def finalize():
        pass
