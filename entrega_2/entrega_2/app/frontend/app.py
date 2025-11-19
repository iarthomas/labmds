# # app/frontend/app.py
# import os, json, requests, gradio as gr

# BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
# TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))

# def ping_backend():
#     try:
#         r = requests.get(f"{BACKEND_URL}/health", timeout=TIMEOUT)
#         r.raise_for_status()
#         return json.dumps(r.json(), indent=2, ensure_ascii=False)
#     except Exception as e:
#         return f"Backend no disponible en {BACKEND_URL}: {e}"

# def reload_model():
#     try:
#         r = requests.post(f"{BACKEND_URL}/model/reload", timeout=TIMEOUT)
#         r.raise_for_status()
#         return json.dumps(r.json(), indent=2, ensure_ascii=False)
#     except Exception as e:
#         return f"No se pudo recargar: {e}"

# def predict_fn(recency_weeks, freq_w_12, num_deliver, num_visit, size,
#                brand, region_id, category, sub_category, segment, package, customer_type):
#     payload = {
#         "records": [{
#             "recency_weeks": recency_weeks,
#             "freq_w_12": freq_w_12,
#             "num_deliver_per_week": num_deliver,
#             "num_visit_per_week": num_visit,
#             "size": size,
#             "brand": brand or "DESCONOCIDO",
#             "region_id": region_id or "DESCONOCIDO",
#             "category": category or "DESCONOCIDO",
#             "sub_category": sub_category or "DESCONOCIDO",
#             "segment": segment or "DESCONOCIDO",
#             "package": package or "DESCONOCIDO",
#             "customer_type": customer_type or "DESCONOCIDO",
#         }]
#     }
#     try:
#         r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=TIMEOUT)
#         if r.status_code == 503:
#             return ("Modelo NO cargado. Copia el .joblib y pulsa 'Recargar modelo'",
#                     "", json.dumps(payload, indent=2, ensure_ascii=False))
#         r.raise_for_status()
#         out = r.json()
#         prob = out["probabilities"][0]
#         pred = out["predictions"][0]
#         cls = "Compra (1)" if pred == 1 else "No compra (0)"
#         return (f"Predicción: {cls}", f"Probabilidad: {prob:.4f}",
#                 json.dumps(out, indent=2, ensure_ascii=False))
#     except Exception as e:
#         return (f"Error llamando backend: {e}", "",
#                 json.dumps(payload, indent=2, ensure_ascii=False))

# with gr.Blocks(title="SodAI Drinks — Predicción semanal") as demo:
#     gr.Markdown(f"""
# # 🥤 SodAI Drinks — Frontend
# Introduce las características del cliente–producto y obtén la **probabilidad de compra la próxima semana**.

# **Cómo usar**
# 1. Pulsa **Ping backend**. Si sale `status: ready`, el modelo está cargado.
# 2. Completa **al menos** `recency_weeks` y `freq_w_12`.
# 3. Pulsa **Predecir**.  
# 4. Si recién copiaste el modelo, usa **Recargar modelo**.

# > Backend: `{BACKEND_URL}`
# """)
#     with gr.Row():
#         ping_btn = gr.Button("Ping backend")
#         reload_btn = gr.Button("Recargar modelo")
#     status_box = gr.Code(label="/health o recarga", lines=10)

#     with gr.Row():
#         recency_weeks = gr.Number(value=2, label="recency_weeks")
#         freq_w_12 = gr.Number(value=5, label="freq_w_12")
#         size = gr.Number(value=1.5, label="size")
#     with gr.Row():
#         num_deliver = gr.Number(value=1, label="num_deliver_per_week")
#         num_visit = gr.Number(value=1, label="num_visit_per_week")
#     with gr.Accordion("Categóricas (opcional)", open=False):
#         with gr.Row():
#             brand = gr.Textbox(value="BR1", label="brand")
#             region_id = gr.Textbox(value="RM", label="region_id")
#             category = gr.Textbox(value="Bebidas", label="category")
#         with gr.Row():
#             sub_category = gr.Textbox(value="Gaseosa", label="sub_category")
#             segment = gr.Textbox(value="Regular", label="segment")
#             package = gr.Textbox(value="Botella", label="package")
#             customer_type = gr.Textbox(value="NUEVO", label="customer_type")

#     go = gr.Button("Predecir", variant="primary")
#     out_label = gr.Markdown("Esperando predicción…")
#     out_prob = gr.Markdown(visible=True)
#     out_raw = gr.Code(label="Respuesta JSON", lines=12)

#     ping_btn.click(fn=ping_backend, outputs=status_box)
#     reload_btn.click(fn=reload_model, outputs=status_box)
#     go.click(predict_fn,
#              inputs=[recency_weeks, freq_w_12, num_deliver, num_visit, size,
#                      brand, region_id, category, sub_category, segment, package, customer_type],
#              outputs=[out_label, out_prob, out_raw])

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")))

# app/frontend/app.py
import os, json, requests, gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))
APP_TITLE = "SodAI Drinks — Predicción semanal"

def _safe_json(obj): return json.dumps(obj, indent=2, ensure_ascii=False)

def _get_health():
    r = requests.get(f"{BACKEND_URL}/health", timeout=TIMEOUT); r.raise_for_status(); return r.json()

def _reload_model():
    r = requests.post(f"{BACKEND_URL}/model/reload", timeout=TIMEOUT); r.raise_for_status(); return r.json()

def _post_predict(payload):
    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=TIMEOUT)
    if r.status_code == 503:
        raise RuntimeError("Modelo NO cargado. Copia el .joblib y pulsa 'Recargar modelo'.")
    r.raise_for_status(); return r.json()

def _extract_prob(out_json):
    if "prob_next_week" in out_json: return float(out_json["prob_next_week"])
    if "probabilities" in out_json and out_json["probabilities"]: return float(out_json["probabilities"][0])
    if "probability" in out_json:
        v = out_json["probability"]; 
        return float(v[0] if isinstance(v, (list, tuple)) else v)
    raise KeyError("No encontré la probabilidad en la respuesta del backend.")

def _extract_pred(out_json):
    if "predictions" in out_json and out_json["predictions"]: return int(out_json["predictions"][0])
    if "will_buy" in out_json: return 1 if out_json["will_buy"] else 0
    raise KeyError("No encontré la predicción en la respuesta del backend.")

def ping_backend():
    try:
        return "✅ Backend OK\n" + _safe_json(_get_health())
    except Exception as e:
        return f"❌ Backend no disponible en {BACKEND_URL}\n{e}"

def reload_model():
    try:
        return "♻️ Modelo recargado\n" + _safe_json(_reload_model())
    except Exception as e:
        return f"❌ No se pudo recargar\n{e}"

def predict_fn(recency_weeks, freq_w_12, num_deliver, num_visit, size,
               brand, region_id, category, sub_category, segment, package, customer_type):
    payload = {
        "records": [{
            "recency_weeks": recency_weeks, "freq_w_12": freq_w_12,
            "num_deliver_per_week": num_deliver, "num_visit_per_week": num_visit, "size": size,
            "brand": (brand or "").strip() or "DESCONOCIDO",
            "region_id": (region_id or "").strip() or "DESCONOCIDO",
            "category": (category or "").strip() or "DESCONOCIDO",
            "sub_category": (sub_category or "").strip() or "DESCONOCIDO",
            "segment": (segment or "").strip() or "DESCONOCIDO",
            "package": (package or "").strip() or "DESCONOCIDO",
            "customer_type": (customer_type or "").strip() or "DESCONOCIDO",
        }]
    }
    try:
        out = _post_predict(payload)
        prob = _extract_prob(out); pred = _extract_pred(out); thr = float(out.get("threshold", 0.5))
        label = "Compra (1)" if pred == 1 else "No compra (0)"
        html = f"""
        <div style="border:1px solid #2b364b;border-radius:14px;padding:14px">
          <div style="font-weight:700;margin-bottom:8px">Resultado</div>
          <div style="display:inline-block;padding:6px 10px;border-radius:999px;
                      color:{'#10b981' if pred==1 else '#ef4444'};
                      border:1px solid {'#10b98133' if pred==1 else '#ef444433'};
                      background:{'#10b9811a' if pred==1 else '#ef44441a'}">{label}</div>
          <div style="margin-top:10px">
            <div style="position:relative;height:12px;background:#0f172a;border:1px solid #1f2937;border-radius:8px;overflow:hidden">
              <div style="position:absolute;left:0;top:0;bottom:0;background:#f59e0b;width:{prob*100:.2f}%"></div>
              <div style="position:absolute;top:-2px;bottom:-2px;width:2px;background:#93c5fd;left:{thr*100:.2f}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:12px;opacity:.8">
              <span>0</span><span>{prob:.4f}</span><span>1</span>
            </div>
          </div>
        </div>
        """
        return html, out
    except Exception as e:
        return f"""<div style="border:1px solid #ef4444;border-radius:14px;padding:12px;background:#2a0f12">
        <b>Error</b><br>{str(e)}<details><summary>Payload</summary><pre>{_safe_json(payload)}</pre></details></div>""", {}

# Fallbacks para compatibilidad de componentes
HAS_JSON = hasattr(gr, "JSON")
def DropdownCompat(*args, **kwargs):
    try: return gr.Dropdown(*args, **kwargs)
    except TypeError:
        kwargs.pop("allow_custom_value", None); return gr.Dropdown(*args, **kwargs)

try:
    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")
except Exception:
    theme = None  # si la versión no soporta themes

with gr.Blocks(title=APP_TITLE, theme=theme) as demo:
    gr.Markdown(f"### 🥤 {APP_TITLE}\nBackend: `{BACKEND_URL}`")

    with gr.Tab("Predicción"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### Numéricas")
                with gr.Row():
                    recency_weeks = gr.Number(value=2, label="recency_weeks")
                    freq_w_12 = gr.Number(value=5, label="freq_w_12")
                with gr.Row():
                    num_deliver = gr.Number(value=1, label="num_deliver_per_week")
                    num_visit = gr.Number(value=1, label="num_visit_per_week")
                    size = gr.Number(value=1.5, label="size")

                gr.Markdown("#### Categóricas")
                with gr.Row():
                    brand = DropdownCompat(["BR1","BR2","BR3"], value="BR1", label="brand", allow_custom_value=True)
                    region_id = DropdownCompat(["RM","V","VIII","X"], value="RM", label="region_id", allow_custom_value=True)
                    category = DropdownCompat(["Bebidas","Snacks","Lácteos"], value="Bebidas", label="category", allow_custom_value=True)
                with gr.Row():
                    sub_category = DropdownCompat(["Gaseosa","Energética","Agua"], value="Gaseosa", label="sub_category", allow_custom_value=True)
                    segment = DropdownCompat(["Regular","Premium","Económico"], value="Regular", label="segment", allow_custom_value=True)
                    package = DropdownCompat(["Botella","Lata","Pack"], value="Botella", label="package", allow_custom_value=True)
                    customer_type = DropdownCompat(["NUEVO","RECURRENTE","DESCONOCIDO"], value="NUEVO", label="customer_type", allow_custom_value=True)

                go = gr.Button("🚀 Predecir", variant="primary")

            with gr.Column(scale=1):
                result_html = gr.HTML("<div>Llena los campos y presiona <b>Predecir</b>.</div>")
                out_json = gr.JSON(label="Respuesta JSON") if HAS_JSON else gr.Code(label="Respuesta JSON", lines=12)

        gr.Examples(
            examples=[
                [2,5,1,1,1.5,"BR1","RM","Bebidas","Gaseosa","Regular","Botella","NUEVO"],
                [1,9,2,1,2.0,"BR2","V","Bebidas","Energética","Premium","Lata","RECURRENTE"],
            ],
            inputs=[recency_weeks, freq_w_12, num_deliver, num_visit, size,
                    brand, region_id, category, sub_category, segment, package, customer_type],
            label="Ejemplos"
        )

        go.click(
            predict_fn,
            inputs=[recency_weeks, freq_w_12, num_deliver, num_visit, size,
                    brand, region_id, category, sub_category, segment, package, customer_type],
            outputs=[result_html, out_json]
        )

    with gr.Tab("Admin"):
        with gr.Row():
            ping_btn = gr.Button("🩺 Ping backend")
            reload_btn = gr.Button("♻️ Recargar modelo")
        admin_box = gr.Code(label="Estado", lines=12)
        ping_btn.click(fn=ping_backend, outputs=admin_box)
        reload_btn.click(fn=reload_model, outputs=admin_box)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")))
