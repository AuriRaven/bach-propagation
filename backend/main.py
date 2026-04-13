from contextlib import asynccontextmanager

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
    If checkpoint not found, generator is None and /api/generate returns 503.
    """
    await load_generator()
    yield
    # Teardown: nothing required


app = FastAPI(
    title="Motif AI - Bach Propagation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    return {"message": "Bach Propagation API is running"}