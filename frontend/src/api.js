const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

export async function sendChatMessage({ sessionId, message, systemPrompt }) {
  const payload = {
    session_id: sessionId,
    message,
  };
  if (systemPrompt && systemPrompt.trim()) {
    payload.system_prompt = systemPrompt.trim();
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch (networkErr) {
    throw new Error(`Could not reach backend at ${API_BASE}: ${networkErr.message}`);
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Unexpected response from server (HTTP ${response.status})`);
  }

  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : Array.isArray(data?.detail)
          ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : `Request failed with HTTP ${response.status}`;
    const err = new Error(detail);
    err.status = response.status;
    throw err;
  }

  return data;
}
