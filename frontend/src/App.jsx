import { useEffect, useRef, useState, useCallback } from "react";
import DOMPurify from "dompurify";
import "./style.css";

const API_BASE = "http://127.0.0.1:9000";
const createSessionId = () =>
  (crypto?.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`);

const readJson = (key, fallback) => {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
};

const writeJson = (key, value) => {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
};

// API helpers
async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload_pdf`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadPdfs(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_BASE}/upload_pdfs`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function askQuestionWithFiles(query, allowWebSearch, sessionId, fileIds) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, allow_web_search: allowWebSearch, session_id: sessionId, file_ids: fileIds || [] }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchHistory(sessionId) {
  const res = await fetch(`${API_BASE}/history?session_id=${encodeURIComponent(sessionId || "")}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchFiles() {
  const res = await fetch(`${API_BASE}/files`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function deleteSessionApi(sessionId) {
  const res = await fetch(`${API_BASE}/session`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  const initialSessionId = useRef(createSessionId()).current;

  const storedSessions = readJson("sessions", null);
  const storedMessages = readJson("messagesBySession", null);
  const storedCurrentSession = readJson("currentSession", null);

  const [sessions, setSessions] = useState(() =>
    Array.isArray(storedSessions) && storedSessions.length
      ? storedSessions
      : [{ id: initialSessionId, title: "Phiên 1" }]
  );
  const [currentSession, setCurrentSession] = useState(() => {
    if (storedCurrentSession && typeof storedCurrentSession === "string") return storedCurrentSession;
    if (Array.isArray(storedSessions) && storedSessions.length) return storedSessions[0].id;
    return initialSessionId;
  });
  const [messagesBySession, setMessagesBySession] = useState(() => {
    if (storedMessages && typeof storedMessages === "object" && !Array.isArray(storedMessages)) return storedMessages;
    return { [initialSessionId]: [] };
  });
  const [inputStr, setInputStr] = useState("");
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [allowWeb, setAllowWeb] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [files, setFiles] = useState([]);
  const [selectedFileIds, setSelectedFileIds] = useState([]);

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const filesRef = useRef([]);

  const currentMessages = messagesBySession[currentSession] || [];
  const selectedNames = files.filter((f) => selectedFileIds.includes(f.file_id)).map((f) => f.file_name);

  const refreshFiles = useCallback(async () => {
    try {
      const data = await fetchFiles();
      const prevFileIds = new Set(filesRef.current.map((f) => f.file_id));
      filesRef.current = data;
      setFiles(data);
      setSelectedFileIds((prev) => {
        const prevSet = new Set(prev || []);
        const next = data
          .filter((f) => prevSet.has(f.file_id) || !prevFileIds.has(f.file_id))
          .map((f) => f.file_id);
        return next.length ? next : prev;
      });
    } catch (err) {
      console.error("Fetch files failed", err);
    }
  }, []);

  const updateMessages = (sessionId, updater) => {
    setMessagesBySession((prev) => {
      const existing = prev[sessionId] || [];
      const next = typeof updater === "function" ? updater(existing) : updater;
      return { ...prev, [sessionId]: next };
    });
  };

  useEffect(() => {
    if (!sessions.some((s) => s.id === currentSession)) {
      const fallback = sessions[0]?.id || initialSessionId;
      setCurrentSession(fallback);
    }
  }, [sessions, currentSession, initialSessionId]);

  useEffect(() => writeJson("sessions", sessions), [sessions]);
  useEffect(() => writeJson("currentSession", currentSession), [currentSession]);
  useEffect(() => writeJson("messagesBySession", messagesBySession), [messagesBySession]);

  useEffect(() => {
    fetchHistory(currentSession).then(setHistoryList).catch(console.error);
  }, [currentSession]);

  useEffect(() => {
    refreshFiles();
  }, [refreshFiles]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentMessages, loading]);

  const handleNewChat = () => {
    const newId = createSessionId();
    const newTitle = `Phiên ${sessions.length + 1}`;
    setSessions((prev) => [...prev, { id: newId, title: newTitle }]);
    setCurrentSession(newId);
    setMessagesBySession((prev) => ({ ...prev, [newId]: [] }));
    setHistoryList([]);
    setUploadedFile(null);
    setInputStr("");
  };

  const handleSwitchSession = (sessionId) => {
    setCurrentSession(sessionId);
    setInputStr("");
    setUploadedFile(null);
    if (!messagesBySession[sessionId]) {
      setMessagesBySession((prev) => ({ ...prev, [sessionId]: [] }));
    }
  };

  const handleToggleFile = (fileId) => {
    setSelectedFileIds((prev) => {
      const set = new Set(prev || []);
      if (set.has(fileId)) {
        set.delete(fileId);
      } else {
        set.add(fileId);
      }
      return Array.from(set);
    });
  };

  const handleDeleteSession = async (sessionId) => {
    try {
      await deleteSessionApi(sessionId);
    } catch (err) {
      console.error("Delete session failed", err);
    }

    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    setMessagesBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return Object.keys(next).length ? next : { [initialSessionId]: [] };
    });

    if (currentSession === sessionId) {
      const remaining = sessions.filter((s) => s.id !== sessionId);
      const fallbackId = remaining[0]?.id || createSessionId();
      if (!remaining[0]) {
        setSessions([{ id: fallbackId, title: "Phiên 1" }]);
      }
      setCurrentSession(fallbackId);
      setHistoryList([]);
    }
  };

  const handleFileSelect = async (e) => {
    const selected = e.target.files ? Array.from(e.target.files) : [];
    if (!selected.length) return;
    const sessionId = currentSession;

    try {
      setUploading(true);
      if (selected.length === 1) {
        const file = selected[0];
        await uploadPdf(file);
        setUploadedFile(file.name);
        updateMessages(sessionId, (prev) => [...prev, { type: "system", text: `Đã tải lên và xử lý: ${file.name}` }]);
      } else {
        const resp = await uploadPdfs(selected);
        const names = resp.uploaded?.map((f) => f.file_name).join(", ");
        setUploadedFile(names || `${selected.length} files`);
        updateMessages(sessionId, (prev) => [...prev, { type: "system", text: `Đã tải lên: ${names || selected.length + " files"}` }]);
      }
      await refreshFiles();
    } catch (err) {
      updateMessages(sessionId, (prev) => [...prev, { type: "system", text: `Lỗi upload: ${err.message}` }]);
    } finally {
      setUploading(false);
      e.target.value = null;
    }
  };

  const handleSendMessage = async () => {
    if (!inputStr.trim()) return;

    const query = inputStr;
    const sessionId = currentSession;
    setInputStr("");

    updateMessages(sessionId, (prev) => [...prev, { type: "user", text: query }]);
    setLoading(true);

    try {
      const { answer } = await askQuestionWithFiles(query, allowWeb, sessionId, selectedFileIds);
      updateMessages(sessionId, (prev) => [...prev, { type: "bot", text: answer }]);
      const updatedHist = await fetchHistory(sessionId);
      setHistoryList(updatedHist);
    } catch (err) {
      updateMessages(sessionId, (prev) => [...prev, { type: "bot", text: `Lỗi: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInput = (e) => {
    const target = e.target;
    target.style.height = "auto";
    target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
    setInputStr(target.value);
  };

  const renderAnswerHtml = (text) => {
    if (!text) return "";
    const normalizeInline = (line) =>
      line
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');

    const normalized = text.replace(/\r\n/g, "\n").replace(/\u00a0/g, " ").trim();
    const bulletFriendly = normalized.replace(/(^|[\n\r]|[:.])\s*([*-])\s+/g, "\n$2 ");
    const lines = bulletFriendly
      .split("\n")
      .map((ln) => ln.trim())
      .filter(Boolean);

    const htmlParts = [];
    let listBuffer = [];

    const flushList = () => {
      if (!listBuffer.length) return;
      htmlParts.push("<ul>");
      listBuffer.forEach((item) => htmlParts.push(`<li>${normalizeInline(item)}</li>`));
      htmlParts.push("</ul>");
      listBuffer = [];
    };

    for (const line of lines) {
      const bullet = line.match(/^[*-]\s+(.*)/);
      if (bullet) {
        listBuffer.push(bullet[1]);
      } else {
        flushList();
        htmlParts.push(`<p>${normalizeInline(line)}</p>`);
      }
    }
    flushList();

    return DOMPurify.sanitize(htmlParts.join(""));
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="logo">
          <i className="fas fa-atom"></i> RAG COSMIC
        </div>

        <button className="new-chat-btn" onClick={handleNewChat}>
          <i className="fas fa-plus"></i> Cuộc trò chuyện mới
        </button>

        <div className="nav">
          <div className="nav-title">Phiên</div>
          {sessions.map((s) => (
            <div
              key={s.id}
              className="history-item"
              onClick={() => handleSwitchSession(s.id)}
              style={
                s.id === currentSession
                  ? { background: "var(--glass-highlight)", color: "var(--text-primary)", border: "1px solid var(--glass-border)" }
                  : {}
              }
            >
              <i className="far fa-comment-alt"></i>
              <span style={{ flex: 1 }}>{s.title}</span>
              <button
                className="icon-btn"
                title="Xóa phiên"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteSession(s.id);
                }}
                style={{ padding: 6, fontSize: 14 }}
              >
                <i className="fas fa-trash"></i>
              </button>
            </div>
          ))}

          <div className="nav-title">PDF đã tải</div>
          {files.map((f) => {
            const checked = selectedFileIds.includes(f.file_id);
            return (
              <label
                key={f.file_id}
                className="history-item"
                style={{ gap: 8, alignItems: "center", cursor: "pointer" }}
                onClick={() => handleToggleFile(f.file_id)}
              >
                <input type="checkbox" checked={checked} readOnly />
                <span style={{ flex: 1 }}>{f.file_name}</span>
              </label>
            );
          })}
          {files.length === 0 && <div style={{ padding: "0 15px", fontSize: "13px", color: "#64748b" }}>Chưa có PDF</div>}

          <div className="nav-title">Lịch sử phiên này</div>
          {historyList
            .slice()
            .reverse()
            .map((h, idx) => (
              <div key={idx} className="history-item">
                <i className="far fa-clock"></i>
                <span>{h.query}</span>
              </div>
            ))}
          {historyList.length === 0 && <div style={{ padding: "0 15px", fontSize: "13px", color: "#64748b" }}>Chưa có lịch sử</div>}
        </div>

        <div className="profile">
          <div className="avatar">U</div>
          <div style={{ fontSize: "14px", fontWeight: "500" }}>User</div>
        </div>
      </aside>

      <main className="main">
        <div className="chat-scroll-area">
          {currentMessages.length === 0 ? (
            <div className="hero-container">
              <div className="hero-icon">
                <i className="fas fa-robot"></i>
              </div>
              <div className="hero-text">
                <h1>Xin chào, tôi có thể giúp gì?</h1>
                <p>Hệ thống RAG hỗ trợ tra cứu tài liệu PDF và tìm kiếm Web thông minh.</p>
              </div>
            </div>
          ) : (
            currentMessages.map((msg, idx) => {
              const isUser = msg.type === "user";
              const isSystem = msg.type === "system";
              return (
                <div key={idx} className={`message-wrapper ${isUser ? "user" : ""}`}>
                  {!isUser && (
                    <div className="msg-avatar bot">
                      <i className="fas fa-bolt"></i>
                    </div>
                  )}

                  {msg.type === "bot" ? (
                    <div className="msg-content bot-text" dangerouslySetInnerHTML={{ __html: renderAnswerHtml(msg.text) }} />
                  ) : (
                    <div className={`msg-content ${isUser ? "user-text" : "bot-text"}`}>
                      {isSystem ? (
                        <em style={{ color: "#4ade80" }}>
                          <i className="fas fa-check-circle"></i> {msg.text}
                        </em>
                      ) : (
                        msg.text
                      )}
                    </div>
                  )}

                  {isUser && (
                    <div className="msg-avatar user">
                      <i className="fas fa-user"></i>
                    </div>
                  )}
                </div>
              );
            })
          )}
          {loading && (
            <div className="message-wrapper">
              <div className="msg-avatar bot">
                <i className="fas fa-bolt"></i>
              </div>
              <div className="msg-content bot-text" style={{ color: "#94a3b8" }}>
                <i className="fas fa-circle-notch fa-spin"></i> Đang suy nghĩ...
              </div>
            </div>
          )}
          <div ref={chatEndRef}></div>
        </div>

        <div className="input-region">
          <div className="input-container">
            {(selectedNames.length || uploadedFile) && (
              <div className="file-preview">
                <i className="fas fa-file-pdf"></i> {selectedNames.length ? selectedNames.join(", ") : uploadedFile}
                <i
                  className="fas fa-times"
                  style={{ cursor: "pointer", marginLeft: 5 }}
                  onClick={() => {
                    setUploadedFile(null);
                    setSelectedFileIds([]);
                  }}
                ></i>
              </div>
            )}

            <div className="input-row">
              <input
                type="file"
                ref={fileInputRef}
                accept="application/pdf"
                multiple
                style={{ display: "none" }}
                onChange={handleFileSelect}
              />
              <button className="icon-btn" title="Tải lên PDF" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                {uploading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-paperclip"></i>}
              </button>

              <button
                className={`icon-btn ${allowWeb ? "active" : ""}`}
                title={allowWeb ? "Tắt tìm kiếm Web" : "Bật tìm kiếm Web"}
                onClick={() => setAllowWeb(!allowWeb)}
              >
                <i className="fas fa-globe"></i>
              </button>

              <textarea
                rows={1}
                placeholder="Nhập câu hỏi của bạn..."
                value={inputStr}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
              ></textarea>

              <button
                className="icon-btn"
                style={{ color: inputStr ? "#3b82f6" : "inherit" }}
                onClick={handleSendMessage}
                disabled={loading || !inputStr.trim()}
              >
                <i className="fas fa-paper-plane"></i>
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
