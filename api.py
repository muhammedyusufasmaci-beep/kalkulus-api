import base64
import os
import re
from typing import Any

import numpy as np
import sympy as sp
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

app = FastAPI(
    title="Kalkülüs API",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

x = sp.Symbol("x", real=True)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
)

SAFE_LOCALS = {
    "x": x,
    "pi": sp.pi,
    "e": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
}

SAFE_GLOBALS = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Function": sp.Function,
}

ALLOWED_IDENTIFIERS = set(SAFE_LOCALS.keys())


class ExpressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expression: str = Field(min_length=1, max_length=300)


class IntegralRequest(ExpressionRequest):
    lower: float | None = None
    upper: float | None = None
    include_steps: bool = False


class LimitRequest(ExpressionRequest):
    point: float
    direction: str = "both"


def parse_expression(expression: str) -> sp.Expr:
    cleaned = expression.strip().lower()
    cleaned = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")

    if not cleaned:
        raise HTTPException(status_code=400, detail="Matematiksel ifade boş olamaz.")

    if len(cleaned) > 300:
        raise HTTPException(status_code=400, detail="İfade çok uzun.")

    if not re.fullmatch(r"[0-9a-z+\-*/^().,\s]+", cleaned):
        raise HTTPException(
            status_code=400,
            detail="İfade desteklenmeyen karakter içeriyor.",
        )

    identifiers = set(re.findall(r"[a-zA-Z_]+", cleaned))
    unknown = identifiers - ALLOWED_IDENTIFIERS

    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen fonksiyon: {', '.join(sorted(unknown))}",
        )

    try:
        result = parse_expr(
            cleaned,
            local_dict=SAFE_LOCALS,
            global_dict=SAFE_GLOBALS,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

        if result.free_symbols - {x}:
            raise ValueError("Sadece x değişkeni kullanılabilir.")

        return result

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Geçersiz matematiksel ifade.",
        )


def text_result(value: Any) -> str:
    return str(sp.simplify(value)).replace("**", "^")


def sample_points(expr: sp.Expr, x_min: float, x_max: float, n: int = 300):
    try:
        function = sp.lambdify(x, expr, modules=["numpy"])
        xs = np.linspace(x_min, x_max, n)
        ys: list[float | None] = []

        for point in xs:
            try:
                value = function(point)
                number = float(value)

                if not np.isfinite(number) or abs(number) > 1_000_000:
                    ys.append(None)
                else:
                    ys.append(number)
            except Exception:
                ys.append(None)

        return {
            "x": [float(point) for point in xs],
            "y": ys,
        }

    except Exception:
        return None


def make_integral_steps(
    expression: str,
    antiderivative: str,
    lower: float | None = None,
    upper: float | None = None,
) -> list[str]:
    compact = expression.replace(" ", "").replace("**", "^")

    steps = [f"Verilen ifade: ∫({compact}) dx"]

    if compact in {"1/x", "x^-1"}:
        steps.append("Kural: ∫(1/x) dx = ln|x|.")
    elif compact == "sin(x)":
        steps.append("Kural: ∫sin(x) dx = -cos(x).")
    elif compact == "cos(x)":
        steps.append("Kural: ∫cos(x) dx = sin(x).")
    elif compact in {"e^x", "exp(x)"}:
        steps.append("Kural: ∫e^x dx = e^x.")
    elif re.fullmatch(r"[+-]?(\d+\*?)?x(\^\d+)?", compact):
        steps.append(
            "Kuvvet kuralı: ∫x^n dx = x^(n+1)/(n+1), n ≠ -1."
        )
    elif "+" in compact or "-" in compact[1:]:
        steps.append(
            "Toplam/fark kuralı: Her terimin integrali ayrı alınır."
        )
    else:
        steps.append(
            "İfade sembolik integral yöntemiyle sadeleştirilir."
        )

    steps.append(f"İlkel fonksiyon: F(x) = {antiderivative}")

    if lower is not None and upper is not None:
        steps.append(
            f"Temel teorem: ∫[{lower}, {upper}] f(x) dx = F({upper}) - F({lower})."
        )
        steps.append("Alt ve üst sınırlar ilkel fonksiyonda yerine konur.")
    else:
        steps.append("Belirsiz integral olduğu için sonuca + C eklenir.")

    return steps


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Formül okuma servisi yapılandırılmamış.",
        )

    return OpenAI(api_key=api_key)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Kalkülüs API çalışıyor",
        "version": "2.1.0",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/derivative")
def derivative(req: ExpressionRequest):
    expr = parse_expression(req.expression)
    result = sp.simplify(sp.diff(expr, x))

    return {
        "input": req.expression,
        "result": text_result(result),
        "result_latex": sp.latex(result),
        "plot": sample_points(expr, -10, 10),
    }


@app.post("/integral")
def integral(req: IntegralRequest):
    if (req.lower is None) != (req.upper is None):
        raise HTTPException(
            status_code=400,
            detail="Alt ve üst sınır birlikte girilmelidir.",
        )

    expr = parse_expression(req.expression)

    if req.lower is None and req.upper is None:
        result = sp.simplify(sp.integrate(expr, x))

        if result.has(sp.Integral):
            raise HTTPException(
                status_code=422,
                detail="Bu integralin kapalı form çözümü bulunamadı.",
            )

        result_text = text_result(result)

        return {
            "input": req.expression,
            "type": "belirsiz",
            "result": f"{result_text} + C",
            "result_latex": f"{sp.latex(result)} + C",
            "steps": (
                make_integral_steps(req.expression, result_text)
                if req.include_steps
                else []
            ),
            "plot": sample_points(expr, -10, 10),
        }

    result = sp.simplify(sp.integrate(expr, (x, req.lower, req.upper)))

    if result.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise HTTPException(
            status_code=400,
            detail="Bu aralıkta integral sonlu bir değer vermiyor.",
        )

    try:
        numeric_value = float(sp.N(result))
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Bu belirli integral sayısal olarak hesaplanamadı.",
        )

    margin = max(1.0, abs(req.upper - req.lower) * 0.3)
    antiderivative = sp.simplify(sp.integrate(expr, x))

    return {
        "input": req.expression,
        "type": "belirli",
        "lower": req.lower,
        "upper": req.upper,
        "result": text_result(result),
        "numeric_value": numeric_value,
        "steps": (
            make_integral_steps(
                req.expression,
                text_result(antiderivative),
                req.lower,
                req.upper,
            )
            if req.include_steps
            else []
        ),
        "plot": sample_points(expr, req.lower - margin, req.upper + margin),
    }


@app.post("/limit")
def limit(req: LimitRequest):
    if req.direction not in {"left", "right", "both"}:
        raise HTTPException(
            status_code=400,
            detail="Yön left, right veya both olmalıdır.",
        )

    expr = parse_expression(req.expression)

    direction_map = {
        "left": "-",
        "right": "+",
        "both": "+-",
    }

    try:
        result = sp.limit(expr, x, req.point, dir=direction_map[req.direction])
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Limit hesaplanamadı.",
        )

    return {
        "input": req.expression,
        "point": req.point,
        "direction": req.direction,
        "result": text_result(result),
        "plot": sample_points(expr, req.point - 5, req.point + 5),
    }


@app.post("/recognize-formula")
async def recognize_formula(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Sadece JPG, PNG veya WEBP görsel yükleyebilirsin.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Görsel boş.")

    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Görsel en fazla 8 MB olabilir.",
        )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        client = get_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Görseldeki matematiksel ifadeyi dikkatlice oku. "
                                "SADECE matematiksel ifadeyi yaz. Açıklama, cümle, "
                                "Markdown veya kod bloğu kullanma. "
                                "Kesir için \\frac{pay}{payda}, karekök için \\sqrt{x}, "
                                "üs için ^ kullanabilirsin. "
                                "Örnek: x^2+3*x veya \\frac{1}{x^2+1}."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{file.content_type};base64,{image_base64}"
                                )
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=150,
        )

        raw = (response.choices[0].message.content or "").strip()

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Formül okuma servisi şu anda yanıt veremiyor.",
        )

    cleaned = raw.strip("`$ \n")

    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Görselde okunabilir bir matematik ifadesi bulunamadı.",
        )

    # LaTeX/Unicode çıktısını burada reddetmiyoruz.
    # Flutter tarafı bunu hesap makinesi biçimine dönüştürüyor.
    if len(cleaned) > 300:
        raise HTTPException(
            status_code=422,
            detail="Okunan ifade çok uzun.",
        )

    return {"expression": cleaned}
