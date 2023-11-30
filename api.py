import uvicorn
from numpy import mean, argmax
from typing import Annotated
from librosa.feature import mfcc
from fastapi import FastAPI, File
from librosa import load as load_audio
from onnxruntime import InferenceSession


CLASS_NAMES: dict = {
    0: "Ambulance",
    1: "Fire_truck",
    2: "Civilian_car",
}

TIME_DURATION: int = 3


app = FastAPI()
model = InferenceSession("model.onnx")


def procress_audio(file: str):
    audio, sample_rate = load_audio(file, sr=None, mono=True)
    chunk_samples = int(TIME_DURATION * sample_rate)
    audio = audio[:chunk_samples]
    mfccs = mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = mean(mfccs.T, axis=0)
    mfccs_processed = mfccs_processed.reshape(1, -1)
    return mfccs_processed


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: Annotated[bytes, File()]):
    try:
        with open("temp.wav", "wb") as f:
            f.write(file)
        f.close()
    except Exception as e:
        return {"res": "Something went wrong while saving the audio file."}

    try:
        input_sample = procress_audio("temp.wav")
    except Exception as e:
        return {"res": "Something went wrong while processing the audio file."}

    results = model.run(["dense_2"], {"dense_input": input_sample})[0]
    class_index = argmax(results)
    return {"res": CLASS_NAMES[class_index]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5049)
