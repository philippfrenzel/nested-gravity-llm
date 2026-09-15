from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from config import ExperimentConfig
from data.text_data import CharacterTextDataset
from train import make_model


PAGE = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nested Gravity Chat</title>
  <style>
    :root { color-scheme: light; --ink: #16211b; --paper: #f5f0e5; --accent: #006b5e; --line: #bfd0c5; --user: #d7eadb; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: repeating-linear-gradient(0deg, var(--paper), var(--paper) 30px, #eee7d8 31px); color: var(--ink); font: 17px Georgia, serif; }
    main { width: min(760px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 28px; }
    h1 { margin: 0 0 4px; font-size: 30px; font-weight: normal; }
    header p { margin: 0 0 28px; color: #496154; }
    #messages { min-height: 360px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 18px 0; white-space: pre-wrap; }
    .message { margin: 0 0 18px; line-height: 1.5; }
    .label { display: block; margin-bottom: 4px; color: var(--accent); font: 700 12px/1.2 system-ui, sans-serif; letter-spacing: 1px; text-transform: uppercase; }
    .user { padding-left: 14px; border-left: 3px solid var(--user); }
    form { display: grid; grid-template-columns: 1fr auto; gap: 10px; margin-top: 20px; }
    textarea { min-height: 54px; resize: vertical; padding: 12px; border: 1px solid var(--line); border-radius: 4px; background: #fffdf8; color: inherit; font: inherit; }
    button { align-self: end; height: 46px; padding: 0 18px; border: 0; border-radius: 4px; background: var(--accent); color: white; cursor: pointer; font: 700 14px system-ui, sans-serif; }
    button:disabled { opacity: .55; cursor: wait; }
    .error { color: #9b2915; }
    @media (max-width: 540px) { main { width: min(100% - 24px, 760px); padding-top: 24px; } form { grid-template-columns: 1fr; } button { width: 100%; } }
  </style>
</head>
<body>
  <main>
    <header><h1>Nested Gravity Chat</h1><p>Zeichenbasierte Fortsetzung mit dem trainierten Textmodell.</p></header>
    <section id="messages" aria-live="polite"></section>
    <form id="chat-form">
      <textarea id="prompt" aria-label="Nachricht" placeholder="Schreibe einen Anfang ..." required></textarea>
      <button id="send" type="submit">Senden</button>
    </form>
  </main>
  <script>
    const form = document.querySelector('#chat-form');
    const prompt = document.querySelector('#prompt');
    const messages = document.querySelector('#messages');
    const send = document.querySelector('#send');
    function addMessage(label, content, className = '') {
      const item = document.createElement('article');
      item.className = `message ${className}`;
      const heading = document.createElement('span');
      heading.className = 'label';
      heading.textContent = label;
      const text = document.createElement('div');
      text.textContent = content;
      item.append(heading, text);
      messages.append(item);
      item.scrollIntoView({ block: 'end', behavior: 'smooth' });
    }
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const value = prompt.value.trim();
      if (!value) return;
      addMessage('Du', value, 'user');
      prompt.value = '';
      send.disabled = true;
      try {
        const response = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt: value }) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Die Anfrage ist fehlgeschlagen.');
        addMessage('Modell', data.completion);
      } catch (error) {
        addMessage('Fehler', error.message, 'error');
      } finally {
        send.disabled = false;
        prompt.focus();
      }
    });
  </script>
</body>
</html>"""


class TextGenerator:
    def __init__(self, checkpoint_path: Path, temperature: float, max_new_tokens: int) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" not in checkpoint or "model_state" not in checkpoint:
            raise ValueError("Checkpoint must contain 'config' and 'model_state'.")
        self.config = ExperimentConfig(**checkpoint["config"])
        if self.config.task != "text":
            raise ValueError("The chat application requires a checkpoint trained with task='text'.")
        self.tokenizer = CharacterTextDataset.from_path(
            self.config.text_path, self.config.text_sequence_length, self.config.seed
        ).tokenizer
        self.model = make_model(self.config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        if len(self.tokenizer.vocab) != self.config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match the checkpoint. Check text_path in its configuration.")

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Please enter a prompt.")
        unknown = sorted(set(prompt) - set(self.tokenizer.vocab))
        if unknown:
            raise ValueError(f"Unsupported characters: {''.join(unknown)!r}")

        token_ids = self.tokenizer.encode(prompt)
        generated: list[int] = []
        context_length = max(1, self.config.max_sequence_length)
        for _ in range(self.max_new_tokens):
            context = torch.tensor([token_ids[-context_length:]], dtype=torch.long)
            logits = self.model(context)
            distribution = torch.softmax(logits[0, -1] / self.temperature, dim=-1)
            next_token = torch.multinomial(distribution, 1).item()
            token_ids.append(next_token)
            generated.append(next_token)
        return self.tokenizer.decode(generated)


def make_handler(generator: TextGenerator) -> type[BaseHTTPRequestHandler]:
    class ChatHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            page = PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload: dict[str, Any] = json.loads(self.rfile.read(length))
                prompt = payload.get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError("Prompt must be text.")
                self._respond(HTTPStatus.OK, {"completion": generator.generate(prompt)})
            except (ValueError, json.JSONDecodeError) as error:
                self._respond(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _respond(self, status: HTTPStatus, payload: dict[str, str]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ChatHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a small local interface for a text checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/nested_gravity_text_char_best.pt"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    generator = TextGenerator(args.checkpoint, args.temperature, args.max_new_tokens)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(generator))
    print(f"Chat available at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()