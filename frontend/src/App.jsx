import React, { useEffect, useRef, useState } from "react";
import { sendChatMessage } from "./api";

function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 12)}`;
  }
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [systemPrompt, setSystemPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  async function handleSend(e) {
    if (e) e.preventDefault();
    if (isSending) return;

    const textToSend = input;
    setInput("");

    if (!textToSend.trim()) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "error",
          content: "Message cannot be empty.",
        },
      ]);
      return;
    }

    const userMsgId = `u-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: textToSend.trim() },
    ]);
    setIsSending(true);

    try {
      const data = await sendChatMessage({
        sessionId,
        message: textToSend,
        systemPrompt,
      });

      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
    } catch (err) {
      const statusPrefix = err.status ? `Error ${err.status}: ` : "Error: ";
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "error",
          content: `${statusPrefix}${err.message || "Request failed"}`,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleNewSession() {
    setSessionId(generateSessionId());
    setMessages([]);
    setInput("");
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <h1>LLM Gateway Chat</h1>
          <span className="session-pill">Session: {sessionId}</span>
        </div>
        <button
          type="button"
          className="new-session-btn"
          onClick={handleNewSession}
          disabled={isSending}
        >
          New Session
        </button>
      </header>

      <section className="system-prompt-bar">
        <label htmlFor="system-prompt-field">System prompt (optional)</label>
        <input
          id="system-prompt-field"
          data-testid="system-prompt-input"
          type="text"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="Optional system instruction for this session..."
          disabled={isSending}
        />
      </section>

      <main className="chat-feed">
        {messages.length === 0 && (
          <div className="empty-state">
            Send a message to start a multi-turn conversation. Repeat an identical prompt in a new session to see a cache hit.
          </div>
        )}

        {messages.map((msg) => {
          if (msg.role === "user") {
            return (
              <div key={msg.id} className="msg msg-user">
                <div className="msg-label">You</div>
                <div className="msg-text">{msg.content}</div>
              </div>
            );
          }

          if (msg.role === "error") {
            return (
              <div key={msg.id} className="msg msg-error" role="alert">
                <div className="msg-label">Error</div>
                <div className="msg-text">{msg.content}</div>
              </div>
            );
          }

          const meta = msg.metadata || {};
          const isCache = meta.resolution_source === "LLM_CACHE";

          return (
            <div
              key={msg.id}
              className="msg msg-assistant"
              data-testid="assistant-message"
            >
              <div className="msg-label">Assistant</div>
              <div className="msg-text">{msg.content}</div>
              <div className="meta-row">
                <span
                  data-testid="meta-source"
                  className={`badge ${isCache ? "badge-cache" : "badge-live"}`}
                >
                  {meta.resolution_source}
                </span>
                <span className="meta-chip">
                  Model: <strong data-testid="meta-model">{meta.model}</strong>
                </span>
                <span className="meta-chip">
                  Latency:{" "}
                  <strong data-testid="meta-latency">{meta.latency_ms} ms</strong>
                </span>
                <span className="meta-chip">
                  In:{" "}
                  <strong data-testid="meta-input-tokens">
                    {meta.input_tokens}
                  </strong>
                </span>
                <span className="meta-chip">
                  Out:{" "}
                  <strong data-testid="meta-output-tokens">
                    {meta.output_tokens}
                  </strong>
                </span>
                {meta.key_id && (
                  <span className="meta-chip">
                    Key: <strong>{meta.key_id}</strong>
                  </span>
                )}
                {typeof meta.fallback_attempts === "number" &&
                  meta.fallback_attempts > 0 && (
                    <span className="meta-chip badge-warn">
                      Fallbacks: {meta.fallback_attempts}
                    </span>
                  )}
              </div>
            </div>
          );
        })}

        {isSending && <div className="typing-indicator">Thinking...</div>}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={handleSend}>
        <textarea
          data-testid="chat-input"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
          disabled={isSending}
        />
        <button
          type="submit"
          data-testid="send-button"
          className="send-btn"
          disabled={isSending}
        >
          {isSending ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}
