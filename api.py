"""
Mühendislik Ders Uygulaması - Kalkülüs API'si
------------------------------------------------
Türev, integral, limit hesaplamalarını SymPy ile yapan; her sonuç için
grafik noktaları üreten; ve fotoğraftan matematiksel ifade okuyan
(GPT-4o-mini ile) bir web servisi.
"""

import os
import base64

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from openai import OpenAI

# "2x" gibi ifadelerde çarpma işaretini otomatik anlaması için.
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

app = FastAPI(title="Kalkülüs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

x = sp.symbols("x")

# API anahtarını KODA YAZMIYORUZ, Render'da ortam değişkeni olarak
# tanımlayacağız (aşağıda anlatıyorum).
_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def parse_expression(expr_str: str):
    try:
        cleaned = expr_str.replace("^", "**")
        expr = parse_expr(cleaned, local_dict={"x": x}, transformations=_TRANSFORMATIONS)
        return expr
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz matematiksel ifade")


def sample_points(expr, x_min: float, x_max: float, n: int = 300):
    try:
        f = sp.lambdify(x, expr, modules=["numpy"])
    except Exception:
        return None

    xs = np.linspace(x_min, x_max, n)
    ys = []
    for xv in xs:
        try:
            v = f(xv)
            if isinstance(v, complex):
                ys.append(None)
            else:
                fv = float(v)
                if np.isnan(fv) or np.isinf(fv) or abs(fv) > 1e6:
                    ys.append(None)
                else:
                    ys.append(fv)
        except Exception:
            ys.append(None)

    return {"x": [float(v) for v in xs], "y": ys}


class ExpressionRequest(BaseModel):
    expression: str


class IntegralRequest(BaseModel):
    expression: str
    lower: float | None = None
    upper: float | None = None


class LimitRequest(BaseModel):
    expression: str
    point: float
    direction: str = "both"


@app.post("/derivative")
def derivative(req: ExpressionRequest):
    expr = parse_expression(req.expression)
    result = sp.diff(expr, x)
    simplified = sp.simplify(result)
    plot = sample_points(simplified, -10, 10)
    return {
        "input": req.expression,
        "result": str(simplified),
        "result_latex": sp.latex(result),
        "plot": plot,
    }


@app.post("/integral")
def integral(req: IntegralRequest):
    expr = parse_expression(req.expression)

    if req.lower is None or req.upper is None:
        result = sp.integrate(expr, x)
        plot = sample_points(expr, -10, 10)
        return {
            "input": req.expression,
            "type": "belirsiz",
            "result": str(sp.simplify(result)) + " + C",
            "result_latex": sp.latex(result) + " + C",
            "plot": plot,
        }
    else:
        result = sp.integrate(expr, (x, req.lower, req.upper))
        numeric_value = float(sp.N(result))
        margin = max(1.0, (req.upper - req.lower) * 0.3)
        plot = sample_points(expr, req.lower - margin, req.upper + margin)
        return {
            "input": req.expression,
            "type": "belirli",
            "lower": req.lower,
            "upper": req.upper,
            "result": str(result),
            "numeric_value": numeric_value,
            "plot": plot,
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

    plot = sample_points(expr, req.point - 5, req.point + 5)
    return {
        "input": req.expression,
        "point": req.point,
        "direction": req.direction,
        "result": str(result),
        "plot": plot,
    }


@app.post("/recognize-formula")
async def recognize_formula(file: UploadFile = File(...)):
    """Fotoğraftaki matematiksel ifadeyi GPT-4o-mini ile okuyup, uygulamanın
    zaten anladığı düz-metin syntax'ına (x^2 + sin(x) gibi) çevirir."""
    image_bytes = await file.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Bu görseldeki matematiksel ifadeyi oku. SADECE ifadeyi "
                                "düz metin olarak yaz, başka hiçbir şey ekleme (açıklama, "
                                "LaTeX, markdown, ``` işareti yok). Üs için ^ kullan "
                                "(x^2 gibi), çarpma için * ya da bitişik yaz (2x veya 2*x), "
                                "karekök için sqrt(...), fonksiyonlar için sin(x), cos(x), "
                                "tan(x), ln(x), log(x) gibi standart isimler kullan. "
                                "Kesirleri (pay)/(payda) şeklinde parantezli yaz. Tek "
                                "değişken x olduğunu varsay. Örnek doğru çıktı: "
                                "(x^2 + 3*x) / (x - 1)"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=200,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        raise HTTPException(status_code=502, detail="Görsel tanıma servisi şu an cevap vermiyor")

    if not raw:
        raise HTTPException(status_code=422, detail="Görselde bir ifade bulunamadı")

    cleaned = raw.strip("`$ \n")

    # Gerçekten SymPy tarafından anlaşılabiliyor mu diye önden kontrol
    # ediyoruz — anlaşılmazsa kullanıcıya net bir mesaj dönüyoruz.
    try:
        parse_expression(cleaned)
    except HTTPException:
        raise HTTPException(status_code=422, detail=f"Okunan ifade anlaşılamadı: {cleaned}")

    return {"expression": cleaned}


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Kalkülüs API çalışıyor"}
