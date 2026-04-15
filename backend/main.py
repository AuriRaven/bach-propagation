from contextlib import asynccontextmanager
import os # Añadido para manejar variables de entorno dinámicamente

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers.corpus     import corpus_router
from routers.ai_chat    import ai_router
from routers.analysis   import analysis_router
from routers.generation import generation_router, load_generator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — load the music generator once at startup.
    NOTA: En Vercel (Serverless), esto se ejecutará en cada 'cold start'.
    Si el modelo es muy pesado, considera cargarlo bajo demanda (Lazy Load).
    """
    try:
        await load_generator()
    except Exception as e:
        print(f"Error cargando el generador: {e}")
    yield

app = FastAPI(
    title="Motif AI - Bach Propagation",
    lifespan=lifespan,
)

# --- CAMBIO EN CORS ---
# Obtenemos la URL de producción desde una variable de entorno o usamos localhost por defecto
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    # Permitimos tanto local como la URL de producción (o "*" para pruebas)
    allow_origins=[frontend_url, "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(corpus_router,     prefix="/api")
app.include_router(ai_router,         prefix="/api")
app.include_router(analysis_router,   prefix="/api")
app.include_router(generation_router, prefix="/api")

@app.get("/")
def read_root():
    # Útil para verificar que el despliegue fue exitoso desde el navegador
    return {
        "status": "online",
        "project": "Bach Propagation",
        "docs": "/docs"
    }