"""
Mühendislik Ders Uygulaması - Kalkülüs API'si
------------------------------------------------
Türev, integral ve limit hesaplamalarını SymPy ile yapan basit bir web servisi.

Yerel olarak çalıştırmak için:
    pip install fastapi uvicorn sympy
    uvicorn main:app --reload

Sonra tarayıcıdan http://127.0.0.1:8000/docs adresine gidip test edebilirsin.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy as sp

app = FastAPI(title="Kalkülüs API")

# Flutter uygulaması farklı bir "origin"den istek atacağı için CORS'a izin veriyoruz.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

x = sp.symbols("x")


def parse_expression(expr_str: str):
    """Kullanıcının yazdığı metni (örn. 'x^2 + sin(x)') SymPy ifadesine çevirir."""
    try:
        # Kullanıcılar genelde ^ ile üs alışkındır, SymPy ** kullanır.
        cleaned = expr_str.replace("^", "**")
        expr = sp.sympify(cleaned, locals={"x": x})
        return expr
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz matematiksel ifade")


class ExpressionRequest(BaseModel):
    expression: str


class IntegralRequest(BaseModel):
    expression: str
    lower: float | None = None  # boş bırakılırsa belirsiz (sembolik) integral alınır
    upper: float | None = None


class LimitRequest(BaseModel):
    expression: str
    point: float
    direction: str = "both"  # "left", "right", "both"


@app.post("/derivative")
def derivative(req: ExpressionRequest):
    expr = parse_expression(req.expression)
    result = sp.diff(expr, x)
    return {
        "input": req.expression,
        "result": str(sp.simplify(result)),
        "result_latex": sp.latex(result),
    }


@app.post("/integral")
def integral(req: IntegralRequest):
    expr = parse_expression(req.expression)

    if req.lower is None or req.upper is None:
        # Belirsiz integral (sabit C olmadan)
        result = sp.integrate(expr, x)
        return {
            "input": req.expression,
            "type": "belirsiz",
            "result": str(sp.simplify(result)) + " + C",
            "result_latex": sp.latex(result) + " + C",
        }
    else:
        # Belirli integral - SymPy önce sembolik dener, olmazsa sayısala düşer.
        result = sp.integrate(expr, (x, req.lower, req.upper))
        numeric_value = float(sp.N(result))
        return {
            "input": req.expression,
            "type": "belirli",
            "lower": req.lower,
            "upper": req.upper,
            "result": str(result),
            "numeric_value": numeric_value,
        }


@app.post("/limit")
def limit(req: LimitRequest):
    expr = parse_expression(req.expression)

    dir_map = {"left": "-", "right": "+", "both": "+-"}
    sympy_dir = dir_map.get(req.direction, "+-")

    try:
        result = sp.limit(expr, x, req.point, dir=sympy_dir)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Limit hesaplanamadı (fonksiyon bu noktada tanımsız olabilir)",
        )

    return {
        "input": req.expression,
        "point": req.point,
        "direction": req.direction,
        "result": str(result),
    }


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Kalkülüs API çalışıyor"}