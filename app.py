import torch
from models import Generator
from inference import initialize_helper

class InferlessPythonModel:
    def initialize(self):
        initialize_helper('generator_v3')

    def infer(self, inputs):
        audio_url = inputs['audio_url']
        return {"generated_base_64": "This is the result"}

    def finalize():
        pass

