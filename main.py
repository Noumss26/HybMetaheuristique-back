# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers.optimize import router as optimize_router


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Backend livraison-opt démarré")
    yield
    print("🛑 Backend arrêté")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Livraison Optimisation API",
    description=(
        "API d'optimisation d'itinéraires de livraison.\n\n"
        "Algorithmes disponibles : **Ant Colony**, **Genetic**, **Hybrid**, **Local Search**."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS : autorise le frontend Next.js ───────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js dev
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(optimize_router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "message": "Livraison Optimisation API",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}