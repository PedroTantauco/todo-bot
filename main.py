"""
Todo List — API REST + Bot de Telegram con Claude como motor de razonamiento.

Variables de entorno requeridas en Railway:
  DATABASE_URL        → URL de PostgreSQL (Railway la inyecta automáticamente)
  TELEGRAM_TOKEN      → Token del bot obtenido desde @BotFather
  ANTHROPIC_API_KEY   → API key de Anthropic
  API_KEY             → Clave secreta para proteger los endpoints REST
  WEBHOOK_URL         → URL pública de Railway, ej: https://tu-app.railway.app
"""

import os
import json
import logging
import threading

import psycopg2
import psycopg2.extras
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
from telegram import Update, Bot
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATABASE_URL     = os.environ["DATABASE_URL"]
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
API_KEY          = os.environ["API_KEY"]
WEBHOOK_URL      = os.environ["WEBHOOK_URL"]   # ej: https://tu-app.railway.app
PORT             = int(os.environ.get("PORT", 8080))

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─────────────────────────────────────────────
# BASE DE DATOS
# ─────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def init_db():
    """Crea la tabla tasks si no existe."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id         SERIAL PRIMARY KEY,
                    text       TEXT    NOT NULL,
                    status     TEXT    NOT NULL DEFAULT 'todo',
                    priority   INTEGER NOT NULL DEFAULT 5,
                    source     TEXT    NOT NULL DEFAULT 'web',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        conn.commit()
    logger.info("Base de datos lista.")


# ─────────────────────────────────────────────
# FLASK — API REST
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # Permite llamadas desde el frontend en Vercel


def require_api_key(f):
    """Decorador: bloquea requests sin API key válida."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key") or request.args.get("api_key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/tasks", methods=["GET"])
@require_api_key
def get_tasks():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, text, status, priority, source, created_at, updated_at
                FROM tasks
                ORDER BY
                    CASE WHEN status = 'done' THEN 1 ELSE 0 END,
                    priority DESC,
                    created_at ASC
            """)
            rows = cur.fetchall()
    return jsonify([dict(r) for r in rows]), 200


@app.route("/tasks", methods=["POST"])
@require_api_key
def create_task():
    data = request.get_json(force=True)
    text     = data.get("text", "").strip()
    status   = data.get("status", "todo")
    priority = int(data.get("priority", 5))
    source   = data.get("source", "web")

    if not text:
        return jsonify({"error": "El campo 'text' es requerido"}), 400
    if status not in ("todo", "prog", "done"):
        return jsonify({"error": "status debe ser todo, prog o done"}), 400
    if not (1 <= priority <= 10):
        return jsonify({"error": "priority debe estar entre 1 y 10"}), 400

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO tasks (text, status, priority, source)
                VALUES (%s, %s, %s, %s)
                RETURNING id, text, status, priority, source, created_at, updated_at
            """, (text, status, priority, source))
            task = dict(cur.fetchone())
        conn.commit()
    return jsonify(task), 201


@app.route("/tasks/<int:task_id>", methods=["PATCH"])
@require_api_key
def update_task(task_id):
    data = request.get_json(force=True)
    fields = []
    values = []

    if "text" in data:
        fields.append("text = %s")
        values.append(data["text"].strip())
    if "status" in data:
        if data["status"] not in ("todo", "prog", "done"):
            return jsonify({"error": "status debe ser todo, prog o done"}), 400
        fields.append("status = %s")
        values.append(data["status"])
    if "priority" in data:
        p = int(data["priority"])
        if not (1 <= p <= 10):
            return jsonify({"error": "priority debe estar entre 1 y 10"}), 400
        fields.append("priority = %s")
        values.append(p)

    if not fields:
        return jsonify({"error": "Nada que actualizar"}), 400

    fields.append("updated_at = NOW()")
    values.append(task_id)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE id = %s "
                "RETURNING id, text, status, priority, source, created_at, updated_at",
                values,
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        return jsonify({"error": "Tarea no encontrada"}), 404
    return jsonify(dict(row)), 200


@app.route("/tasks/<int:task_id>", methods=["DELETE"])
@require_api_key
def delete_task(task_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tasks WHERE id = %s RETURNING id", (task_id,))
            deleted = cur.fetchone()
        conn.commit()
    if not deleted:
        return jsonify({"error": "Tarea no encontrada"}), 404
    return jsonify({"deleted": task_id}), 200


# ─────────────────────────────────────────────
# CLAUDE — MOTOR DE RAZONAMIENTO
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Eres el asistente de una lista de tareas (to-do list). 
Tu único trabajo es interpretar el mensaje del usuario y devolver una acción JSON.

La lista actual de tareas se incluirá en el mensaje del usuario.

Debes responder ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin comillas de código.

Esquema de respuesta:
{
  "action": "create" | "update" | "delete" | "list" | "unknown",
  "task_id": <número entero o null>,
  "text": <string o null>,
  "status": "todo" | "prog" | "done" | null,
  "priority": <entero 1-10 o null>,
  "reply": <string: mensaje en español para confirmar al usuario>
}

Reglas:
- "create": cuando el usuario quiere agregar una tarea nueva. Extrae texto y prioridad si la menciona.
- "update": cuando quiere cambiar texto, estado o prioridad de una tarea existente. Busca la tarea por nombre aproximado en la lista.
- "delete": cuando quiere eliminar una tarea. Busca la tarea por nombre aproximado.
- "list": cuando quiere ver sus tareas.
- "unknown": cuando no entiendes la intención.
- Si el usuario menciona prioridad como palabra ("urgente", "crítico") → priority 9 o 10. ("baja", "cuando pueda") → 1 a 3.
- status "prog" significa "en progreso".
- El campo "reply" es el mensaje que verá el usuario en Telegram. Sé conciso y amable.
"""


def ask_claude(user_message: str, tasks: list) -> dict:
    """Llama a Claude con el mensaje del usuario y la lista de tareas actual."""
    tasks_summary = "\n".join(
        f"  id={t['id']} | '{t['text']}' | estado={t['status']} | prioridad={t['priority']}"
        for t in tasks
    ) or "  (lista vacía)"

    prompt = f"""Lista de tareas actual:
{tasks_summary}

Mensaje del usuario: \"{user_message}\""""

    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Eliminar posibles bloques de código si Claude los incluye
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ─────────────────────────────────────────────
# TELEGRAM — BOT HANDLERS
# ─────────────────────────────────────────────

def fetch_tasks_internal():
    """Obtiene tareas directamente desde la BD (llamada interna, sin HTTP)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, text, status, priority
                FROM tasks
                ORDER BY
                    CASE WHEN status = 'done' THEN 1 ELSE 0 END,
                    priority DESC,
                    created_at ASC
            """)
            return [dict(r) for r in cur.fetchall()]


def execute_action(action: dict) -> str:
    """Ejecuta la acción decidida por Claude sobre la BD."""
    a = action.get("action")

    if a == "list":
        tasks = fetch_tasks_internal()
        if not tasks:
            return "Tu lista está vacía. ¡Agrega una tarea!"
        STATUS_LABELS = {"todo": "📋 Por Hacer", "prog": "⚙️ En Progreso", "done": "✅ Hecho"}
        lines = []
        for t in tasks:
            label = STATUS_LABELS.get(t["status"], t["status"])
            lines.append(f"[{t['id']}] P{t['priority']} — {t['text']}  ({label})")
        return "Tus tareas:\n\n" + "\n".join(lines)

    if a == "create":
        text     = action.get("text") or ""
        status   = action.get("status") or "todo"
        priority = action.get("priority") or 5
        if not text:
            return "No entendí el nombre de la tarea. ¿Puedes repetirlo?"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO tasks (text, status, priority, source) VALUES (%s, %s, %s, 'bot') RETURNING id",
                    (text, status, int(priority)),
                )
                new_id = cur.fetchone()[0]
            conn.commit()
        return action.get("reply") or f"✅ Tarea #{new_id} creada: \"{text}\""

    if a == "update":
        task_id = action.get("task_id")
        if not task_id:
            return action.get("reply") or "No encontré esa tarea en tu lista."
        fields, values = [], []
        if action.get("text"):
            fields.append("text = %s"); values.append(action["text"])
        if action.get("status"):
            fields.append("status = %s"); values.append(action["status"])
        if action.get("priority"):
            fields.append("priority = %s"); values.append(int(action["priority"]))
        if not fields:
            return "No entendí qué cambiar de la tarea."
        fields.append("updated_at = NOW()")
        values.append(int(task_id))
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE tasks SET {', '.join(fields)} WHERE id = %s RETURNING id",
                    values,
                )
                updated = cur.fetchone()
            conn.commit()
        if not updated:
            return f"No encontré la tarea #{task_id}."
        return action.get("reply") or f"✏️ Tarea #{task_id} actualizada."

    if a == "delete":
        task_id = action.get("task_id")
        if not task_id:
            return action.get("reply") or "No encontré esa tarea en tu lista."
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tasks WHERE id = %s RETURNING id", (int(task_id),))
                deleted = cur.fetchone()
            conn.commit()
        if not deleted:
            return f"No encontré la tarea #{task_id}."
        return action.get("reply") or f"🗑️ Tarea #{task_id} eliminada."

    # unknown
    return action.get("reply") or "No entendí tu mensaje. Puedes decirme cosas como:\n• \"agrega comprar pan prioridad 7\"\n• \"mueve el dentista a en progreso\"\n• \"elimina la tarea de gym\"\n• \"muéstrame mis tareas\""


def handle_start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "¡Hola! Soy tu asistente de tareas 📋\n\n"
        "Puedes decirme cosas como:\n"
        "• \"agrega llamar al médico prioridad 8\"\n"
        "• \"mueve gym a en progreso\"\n"
        "• \"elimina la tarea de compras\"\n"
        "• \"muéstrame mis tareas\"\n\n"
        "¿En qué te ayudo?"
    )


def handle_message(update: Update, context: CallbackContext):
    user_text = update.message.text
    logger.info(f"Mensaje recibido: {user_text}")

    try:
        tasks  = fetch_tasks_internal()
        action = ask_claude(user_text, tasks)
        logger.info(f"Acción Claude: {action}")
        reply  = execute_action(action)
    except json.JSONDecodeError as e:
        logger.error(f"Claude devolvió JSON inválido: {e}")
        reply = "Hubo un problema interpretando tu mensaje. Inténtalo de nuevo."
    except Exception as e:
        logger.error(f"Error procesando mensaje: {e}")
        reply = "Ocurrió un error inesperado. Inténtalo en un momento."

    update.message.reply_text(reply)


# ─────────────────────────────────────────────
# ARRANQUE
# ─────────────────────────────────────────────

def start_bot():
    """Inicia el bot de Telegram en un hilo separado usando webhook."""
    bot = Bot(token=TELEGRAM_TOKEN)
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", handle_start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    webhook_url = f"{WEBHOOK_URL}/webhook/{TELEGRAM_TOKEN}"
    bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook configurado en: {webhook_url}")

    updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=f"/webhook/{TELEGRAM_TOKEN}",
        webhook_url=webhook_url,
    )
    logger.info("Bot de Telegram iniciado.")


@app.route(f"/webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    """Flask recibe los updates de Telegram y los pasa al bot."""
    if token != TELEGRAM_TOKEN:
        return jsonify({"error": "Forbidden"}), 403
    update = Update.de_json(request.get_json(force=True), Bot(token=TELEGRAM_TOKEN))
    # Procesar en hilo separado para no bloquear Flask
    threading.Thread(
        target=lambda: Updater(token=TELEGRAM_TOKEN, use_context=True).dispatcher.process_update(update)
    ).start()
    return "ok", 200


if __name__ == "__main__":
    init_db()
    start_bot()
    # Flask sirve tanto la API REST como el webhook de Telegram
    app.run(host="0.0.0.0", port=PORT)
