from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from Data_Preprocessing import convert_survey_excel_to_csv
from Retrieval_Augmented_Generation import Rag_model
import os, shutil
import warnings
warnings.filterwarnings('ignore')
warnings.warn('Error: A warning just appeared')

app = FastAPI()
model = Rag_model().rag()

# Simulate the backend function processing
async def backend_function():
    try:
        preprocessing_status = convert_survey_excel_to_csv().read_data()
        if preprocessing_status:
            # Replace this with your actual logic
            return {"status": "success", "message": "Data Preprocessed sucessfully"}
        else:
            return {"status": "success", "message": f"{preprocessing_status}"}
    except Exception as e:
        return {"status": "Error", "message": f"{e}"}


# Endpoint to trigger the backend function
@app.post("/run-function/")
async def trigger_function():
    result = await backend_function()
    return JSONResponse(result)


@app.post("/upload-files/")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        convert_survey_excel_to_csv().verify_folder(r"Input")
        for file in files:
            file_location = os.path.join(r"Input", file.filename)
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
        return JSONResponse(content={"files": "saved sucessfully"})
    except Exception as e:
        print(e)
        return JSONResponse(content={"files": f"{e}"})

class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(question: Question):

    try:
        retrieved_output = model.invoke({"query": question.question})
        return JSONResponse({"query": question.question, "response": retrieved_output["result"]})
    except Exception as e:
        print(e)
        return JSONResponse({"query": question.question, "response": e})

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")