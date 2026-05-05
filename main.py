import os
import json
import logging
import psycopg2
import psycopg2.extras
from flask import Flask, request
from flask_cors import CORS
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters, CallbackContext
import anthropic

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DATABASE_URL     = os.environ["DATABASE_URL"]
WEBHOOK_URL      = os.environ.get("WEBHOOK_URL", "")   # e.g. https://your-app.railway.app
PORT             = int(os.environ.get("PORT", 8080))

# ── Clients ────────────────────────────────────────────────────────────────────
bot              = Bot(token=TELEGRAM_TOKEN)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = Flask(__name__)
CORS(app)

# ── Database ───────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    """Create the tasks table if it doesn't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id        SERIAL PRIMARY KEY,
                    text      TEXT        NOT NULL,
                    status    VARCHAR(20) NOT NULL DEFAULT 'todo',
                    priority  INTEGER     NOT NULL DEFAULT 5,
                    created   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        conn.commit()
    logger.info("Database ready.")

def db_get_tasks():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM tasks ORDER BY priority DESC, created DESC")
            return cur.fetchall()

def db_add_task(text: str, priority: int = 5) -> dict:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO tasks (text, status, priority) VALUES (%s, 'todo', %s) RETURNING *",
                (text, priority)
            )
            row = cur.fetchone()
        conn.commit()
    return dict(row)

def db_delete_task(task_id: int) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted

def db_update_task(task_id: int, text: str = None, status: str = None, priority: int = None) -> dict | None:
    fields, values = [], []
    if text     is not None: fields.append("text = %s");     values.append(text)
    if status   is not None: fields.append("status = %s");   values.append(status)
    if priority is not None: fields.append("priority = %s"); values.append(priority)
    if not fields:
        return None
    values.append(task_id)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE id = %s RETURNING *",
                values
            )
            row = cur.fetchone()
        conn.commit()
    return dict(row) if row else None

# ── Anthropic NLP ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Eres el asistente de una lista de tareas Kanban. El usuario te enviará instrucciones en lenguaje natural en español.
Tu única función es convertir esas instrucciones en una acción estructurada en JSON.

Acciones disponibles:
- add:    agregar una nueva tarea
- delete: eliminar una tarea existente por su ID
- edit:   modificar el texto, estado o prioridad de una tarea existente por su ID
- list:   listar todas las tareas actuales
- move:   mover una tarea a otro estado (equivale a edit con status)

Estados válidos: "todo", "prog", "done"
Prioridad: número entero del 1 al 10 (10 = más urgente). Si no se menciona, usa 5.

Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin markdown, sin comillas de bloque.

Esquema de respuesta:
{
  "action": "add" | "delete" | "edit" | "list" | "move",
  "id": <número, solo si aplica>,
  "text": "<texto de la tarea, solo si aplica>",
  "priority": <número 1-10, solo si aplica>,
  "status": "<todo|prog|done, solo si aplica>",
  "clarification": "<mensaje si la instrucción es ambigua o falta información>"
}

Ejemplos:
- "agrega comprar leche con prioridad 8"
  → {"action":"add","text":"comprar leche","priority":8}

- "elimina la tarea 3"
  → {"action":"delete","id":3}

- "mueve la tarea 5 a en progreso"
  → {"action":"move","id":5,"status":"prog"}

- "cambia el texto de la tarea 2 a preparar informe"
  → {"action":"edit","id":2,"text":"preparar informe"}

- "muestra mis tareas"
  → {"action":"list"}

Si la instrucción no corresponde a ninguna acción válida, responde:
{"action":"unknown","clarification":"<explicación breve de qué no se entendió>"}
"""

def interpret_message(user_text: str, task_context: str) -> dict:
    """Send user message to Claude and get back a structured action."""
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Tareas actuales en la base de datos:\n{task_context}\n\nInstrucción del usuario: {user_text}"
                }
            ]
        )
        raw = response.content[0].text.strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Claude returned non-JSON: %s", raw)
        return {"action": "unknown", "clarification": "No pude interpretar la respuesta del modelo."}
    except Exception as e:
        logger.error("Anthropic error: %s", e)
        return {"action": "unknown", "clarification": f"Error al contactar el modelo: {e}"}

def build_task_context() -> str:
    """Build a compact text summary of current tasks for Claude's context."""
    try:
        tasks = db_get_tasks()
        if not tasks:
            return "No hay tareas en este momento."
        lines = []
        for t in tasks:
            status_label = {"todo": "Por hacer", "prog": "En progreso", "done": "Hecho"}.get(t["status"], t["status"])
            lines.append(f"ID {t['id']}: [{status_label}] P{t['priority']} — {t['text']}")
        return "\n".join(lines)
    except Exception:
        return "No se pudo obtener el listado de tareas."

# ── Telegram handlers ──────────────────────────────────────────────────────────
STATUS_EMOJI = {"todo": "🔵", "prog": "🟡", "done": "✅"}
STATUS_LABEL = {"todo": "Por hacer", "prog": "En progreso", "done": "Hecho"}

def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "👋 Hola. Soy tu asistente de tareas.\n\n"
        "Puedes escribirme instrucciones en español como:\n"
        "• *agrega revisar el informe con prioridad 7*\n"
        "• *mueve la tarea 3 a en progreso*\n"
        "• *elimina la tarea 5*\n"
        "• *muéstrame mis tareas*\n\n"
        "Usa /lista para ver todas tus tareas en cualquier momento.",
        parse_mode="Markdown"
    )

def cmd_lista(update: Update, context: CallbackContext):
    try:
        tasks = db_get_tasks()
        if not tasks:
            update.message.reply_text("No tienes tareas registradas aún.")
            return
        by_status = {"todo": [], "prog": [], "done": []}
        for t in tasks:
            by_status.get(t["status"], by_status["todo"]).append(t)
        lines = ["📋 *Tus tareas:*\n"]
        for status in ["todo", "prog", "done"]:
            group = by_status[status]
            if not group:
                continue
            lines.append(f"{STATUS_EMOJI[status]} *{STATUS_LABEL[status]}*")
            for t in group:
                lines.append(f"  `ID {t['id']}` P{t['priority']} — {t['text']}")
            lines.append("")
        update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.error("cmd_lista error: %s", e)
        update.message.reply_text("Error al obtener las tareas.")

def handle_message(update: Update, context: CallbackContext):
    user_text = update.message.text.strip()
    if not user_text:
        return

    task_context = build_task_context()
    action_data  = interpret_message(user_text, task_context)
    action       = action_data.get("action", "unknown")

    try:
        if action == "add":
            text     = action_data.get("text", "").strip()
            priority = int(action_data.get("priority", 5))
            if not text:
                update.message.reply_text("No entendí el texto de la tarea. ¿Puedes repetirlo?")
                return
            priority = max(1, min(10, priority))
            task = db_add_task(text, priority)
            update.message.reply_text(
                f"✅ Tarea creada\n`ID {task['id']}` · P{task['priority']}\n_{task['text']}_",
                parse_mode="Markdown"
            )

        elif action == "delete":
            task_id = int(action_data.get("id", 0))
            if not task_id:
                update.message.reply_text("¿Qué ID de tarea quieres eliminar?")
                return
            ok = db_delete_task(task_id)
            if ok:
                update.message.reply_text(f"🗑 Tarea `ID {task_id}` eliminada.", parse_mode="Markdown")
            else:
                update.message.reply_text(f"No encontré una tarea con ID {task_id}.")

        elif action in ("edit", "move"):
            task_id  = int(action_data.get("id", 0))
            text     = action_data.get("text")
            status   = action_data.get("status")
            priority = action_data.get("priority")
            if not task_id:
                update.message.reply_text("¿Qué ID de tarea quieres modificar?")
                return
            if priority is not None:
                priority = max(1, min(10, int(priority)))
            task = db_update_task(task_id, text=text, status=status, priority=priority)
            if task:
                status_label = STATUS_LABEL.get(task["status"], task["status"])
                update.message.reply_text(
                    f"✏️ Tarea actualizada\n`ID {task['id']}` · {STATUS_EMOJI.get(task['status'], '')} {status_label} · P{task['priority']}\n_{task['text']}_",
                    parse_mode="Markdown"
                )
            else:
                update.message.reply_text(f"No encontré una tarea con ID {task_id}.")

        elif action == "list":
            cmd_lista(update, context)

        elif action == "unknown":
            clarification = action_data.get("clarification", "No entendí la instrucción.")
            update.message.reply_text(f"🤔 {clarification}\n\nEscribe /lista para ver tus tareas actuales.")

        else:
            update.message.reply_text("No reconocí esa acción. Intenta de nuevo o escribe /lista.")

    except Exception as e:
        logger.error("handle_message error: %s", e)
        update.message.reply_text("Ocurrió un error al ejecutar la acción. Intenta de nuevo.")

# ── Flask routes ───────────────────────────────────────────────────────────────
dispatcher = Dispatcher(bot, None, use_context=True)
dispatcher.add_handler(CommandHandler("start", cmd_start))
dispatcher.add_handler(CommandHandler("lista", cmd_lista))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

@app.route(f"/webhook/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    data   = request.get_json(force=True)
    update = Update.de_json(data, bot)
    dispatcher.process_update(update)
    return "ok", 200

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    if not WEBHOOK_URL:
        return {"error": "WEBHOOK_URL env var not set"}, 400
    url = f"{WEBHOOK_URL}/webhook/{TELEGRAM_TOKEN}"
    result = bot.set_webhook(url=url)
    return {"webhook_set": result, "url": url}, 200

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    if WEBHOOK_URL:
        webhook_url = f"{WEBHOOK_URL}/webhook/{TELEGRAM_TOKEN}"
        bot.set_webhook(url=webhook_url)
        logger.info("Webhook set to %s", webhook_url)
    else:
        logger.warning("WEBHOOK_URL not set — webhook not registered. Call /set_webhook after deploy.")
    app.run(host="0.0.0.0", port=PORT)
