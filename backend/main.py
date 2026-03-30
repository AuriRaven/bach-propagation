from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Esta es la variable "app" que Uvicorn está buscando
app = FastAPI(title="Motif AI - Bach Propagation")

# Configuración de CORS para que tu frontend en el puerto 3000 pueda hablar con el backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Bach Propagation API is running"}