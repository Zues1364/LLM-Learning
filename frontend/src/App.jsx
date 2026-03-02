
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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

async function askQuestionWithFiles(query, allowWebSearch, sessionId, fileIds, programId) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      allow_web_search: allowWebSearch,
      session_id: sessionId,
      file_ids: fileIds || [],
      program_id: programId || null,
    }),
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

// --- Resource API ---
async function fetchResources() {
  const res = await fetch(`${API_BASE}/api/resources`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchPrograms(refresh = false) {
  const url = refresh ? `${API_BASE}/api/programs?refresh=true` : `${API_BASE}/api/programs`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadResourcePdf(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/resources/pdf`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadResourcePdfs(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_BASE}/api/resources/pdfs`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadResourceHtml(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/resources/html`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadResourceHtmls(files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_BASE}/api/resources/htmls`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function addResourceUrl(url) {
  const res = await fetch(`${API_BASE}/api/resources/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function deleteResource(id) {
  const res = await fetch(`${API_BASE}/api/resources/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  const initialSessionId = useRef(createSessionId()).current;

  const storedSessions = readJson("sessions", null);
  const storedMessages = readJson("messagesBySession", null);
  const storedCurrentSession = readJson("currentSession", null);
  const storedSelectedPrograms = readJson("selectedProgramBySession", null);
  const storedPendingPrograms = readJson("pendingProgramBySession", null);
  const storedSelectedFiles = readJson("selectedFilesBySession", null);

  const [sessions, setSessions] = useState(() =>
    Array.isArray(storedSessions) && storedSessions.length
      ? storedSessions
      : [{ id: initialSessionId, title: "Phien 1" }]
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
  const [selectedFilesBySession, setSelectedFilesBySession] = useState(() => {
    if (
      storedSelectedFiles &&
      typeof storedSelectedFiles === "object" &&
      !Array.isArray(storedSelectedFiles)
    ) {
      return storedSelectedFiles;
    }
    return {};
  });
  const [processingPdf, setProcessingPdf] = useState(false);
  const [processingLabel, setProcessingLabel] = useState("");

  // Resource State
  const [resources, setResources] = useState([]);
  const [showResourcePanel, setShowResourcePanel] = useState(false);
  const [resourceUrl, setResourceUrl] = useState("");
  const [resourceLoading, setResourceLoading] = useState(false);
  const [programs, setPrograms] = useState([]);
  const [programsLoading, setProgramsLoading] = useState(false);
  const [selectedProgramBySession, setSelectedProgramBySession] = useState(() => {
    if (
      storedSelectedPrograms &&
      typeof storedSelectedPrograms === "object" &&
      !Array.isArray(storedSelectedPrograms)
    ) {
      return storedSelectedPrograms;
    }
    return {};
  });
  const [pendingProgramBySession, setPendingProgramBySession] = useState(() => {
    if (
      storedPendingPrograms &&
      typeof storedPendingPrograms === "object" &&
      !Array.isArray(storedPendingPrograms)
    ) {
      return storedPendingPrograms;
    }
    return {};
  });

  const fileInputRef = useRef(null);
  const resourceFileInputRef = useRef(null);
  const resourceHtmlInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const filesRef = useRef([]);

  const normalizeFileIds = useCallback((ids) => Array.from(new Set((ids || []).filter(Boolean))), []);

  const currentMessages = messagesBySession[currentSession] || [];
  const selectedFileIds = selectedFilesBySession[currentSession] || [];
  const selectedNames = files.filter((f) => selectedFileIds.includes(f.file_id)).map((f) => f.file_name);
  const visibleFiles = files;
  const currentSelectedProgramId = selectedProgramBySession[currentSession] || "";
  const currentPendingProgramId =
    pendingProgramBySession[currentSession] || currentSelectedProgramId || programs[0]?.id || "";
  const currentProgramDisplayName =
    programs.find((p) => p.id === currentSelectedProgramId)?.display_name || currentSelectedProgramId || "";
  const groupedPrograms = useMemo(() => {
    const groups = new Map();

    const asYear = (value) => {
      const n = Number(value);
      return Number.isFinite(n) ? n : -1;
    };

    for (const program of programs) {
      const groupName = String(program?.group_name || program?.name || "Khác").trim() || "Khác";
      if (!groups.has(groupName)) groups.set(groupName, []);
      groups.get(groupName).push(program);
    }

    const collator = new Intl.Collator("vi", { sensitivity: "base" });
    const sortedGroupNames = Array.from(groups.keys()).sort((a, b) => collator.compare(a, b));

    return sortedGroupNames.map((groupName) => {
      const items = (groups.get(groupName) || []).slice().sort((a, b) => {
        const sortYearA = asYear(a?.year_end ?? a?.year);
        const sortYearB = asYear(b?.year_end ?? b?.year);
        if (sortYearA !== sortYearB) return sortYearB - sortYearA;
        const yearA = asYear(a?.year);
        const yearB = asYear(b?.year);
        if (yearA !== yearB) return yearB - yearA;
        return String(a?.id || "").localeCompare(String(b?.id || ""));
      });
      return { groupName, items };
    });
  }, [programs]);

  const updateSelectedFiles = useCallback((sessionId, updater) => {
    setSelectedFilesBySession((prev) => {
      const existing = prev[sessionId] || [];
      const nextRaw = typeof updater === "function" ? updater(existing) : updater;
      return { ...prev, [sessionId]: normalizeFileIds(nextRaw) };
    });
  }, [normalizeFileIds]);

  const handleSelectAllFiles = () => updateSelectedFiles(currentSession, files.map((f) => f.file_id));

  const refreshFiles = useCallback(async () => {
    try {
      const data = await fetchFiles();
      filesRef.current = data;
      setFiles(data);
      const validIds = new Set(data.map((f) => f.file_id));
      setSelectedFilesBySession((prev) => {
        const next = {};
        Object.entries(prev || {}).forEach(([sessionId, ids]) => {
          next[sessionId] = (ids || []).filter((id) => validIds.has(id));
        });
        return next;
      });
    } catch (err) {
      console.error("Fetch files failed", err);
    }
  }, []);

  const refreshResources = useCallback(async () => {
    try {
      const data = await fetchResources();
      setResources(data);
    } catch (err) {
      console.error("Fetch resources failed", err);
    }
  }, []);

  const refreshPrograms = useCallback(async (refresh = false) => {
    try {
      setProgramsLoading(true);
      const data = await fetchPrograms(refresh);
      const list = Array.isArray(data?.programs) ? data.programs : [];
      setPrograms(list);
    } catch (err) {
      console.error("Fetch programs failed", err);
      setPrograms([]);
    } finally {
      setProgramsLoading(false);
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
  useEffect(() => writeJson("selectedProgramBySession", selectedProgramBySession), [selectedProgramBySession]);
  useEffect(() => writeJson("pendingProgramBySession", pendingProgramBySession), [pendingProgramBySession]);
  useEffect(() => writeJson("selectedFilesBySession", selectedFilesBySession), [selectedFilesBySession]);

  useEffect(() => {
    fetchHistory(currentSession).then(setHistoryList).catch(console.error);
  }, [currentSession]);

  useEffect(() => {
    refreshFiles();
    refreshResources();
    refreshPrograms(false);
  }, [refreshFiles, refreshResources, refreshPrograms]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentMessages, loading]);

  const handleNewChat = () => {
    const newId = createSessionId();
    const newTitle = `Phien ${sessions.length + 1}`;
    setSessions((prev) => [...prev, { id: newId, title: newTitle }]);
    setCurrentSession(newId);
    setMessagesBySession((prev) => ({ ...prev, [newId]: [] }));
    setSelectedFilesBySession((prev) => ({ ...prev, [newId]: [] }));
    setPendingProgramBySession((prev) => ({ ...prev, [newId]: programs[0]?.id || "" }));
    setHistoryList([]);
    setUploadedFile(null);
    setProcessingPdf(false);
    setProcessingLabel("");
    setInputStr("");
  };

  const handleSwitchSession = (sessionId) => {
    setCurrentSession(sessionId);
    setInputStr("");
    setUploadedFile(null);
    setProcessingPdf(false);
    setProcessingLabel("");
    if (!messagesBySession[sessionId]) {
      setMessagesBySession((prev) => ({ ...prev, [sessionId]: [] }));
    }
    if (!pendingProgramBySession[sessionId] && !selectedProgramBySession[sessionId] && programs[0]?.id) {
      setPendingProgramBySession((prev) => ({ ...prev, [sessionId]: programs[0].id }));
    }
  };

  const handleToggleFile = (fileId) => {
    updateSelectedFiles(currentSession, (prev) => {
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
    setSelectedProgramBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    setSelectedFilesBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    setPendingProgramBySession((prev) => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });

    if (currentSession === sessionId) {
      const remaining = sessions.filter((s) => s.id !== sessionId);
      const fallbackId = remaining[0]?.id || createSessionId();
      if (!remaining[0]) {
        setSessions([{ id: fallbackId, title: "Phien 1" }]);
      }
      setCurrentSession(fallbackId);
      setHistoryList([]);
    }
  };

  const handleRenameSession = (sessionId) => {
    const target = sessions.find((s) => s.id === sessionId);
    if (!target) return;
    const nextTitle = window.prompt("Nhap ten phien", target.title)?.trim();
    if (!nextTitle) return;
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? { ...s, title: nextTitle } : s)));
  };

  const handleFileSelect = async (e) => {
    const selected = e.target.files ? Array.from(e.target.files) : [];
    if (!selected.length) return;
    const sessionId = currentSession;

    try {
      setUploading(true);
      if (selected.length === 1) {
        const file = selected[0];
        const resp = await uploadPdf(file);
        updateSelectedFiles(sessionId, (prev) => {
          const set = new Set(prev || []);
          set.add(resp.file_id);
          return Array.from(set);
        });
        setUploadedFile(file.name);
      } else {
        const resp = await uploadPdfs(selected);
        const names = resp.uploaded?.map((f) => f.file_name).join(", ");
        const newIds = resp.uploaded?.map((f) => f.file_id).filter(Boolean) || [];
        setUploadedFile(names || `${selected.length} files`);
        updateSelectedFiles(sessionId, (prev) => {
          const set = new Set(prev || []);
          newIds.forEach((id) => set.add(id));
          return Array.from(set);
        });
      }
      await refreshFiles();
    } catch (err) {
      updateMessages(sessionId, (prev) => [...prev, { type: "system", text: `Loi upload: ${err.message}` }]);
    } finally {
      setUploading(false);
      e.target.value = null;
    }
  };

  const handleResourceUpload = async (e) => {
    const selected = e.target.files ? Array.from(e.target.files) : [];
    if (!selected.length) return;
    setResourceLoading(true);
    try {
      if (selected.length === 1) {
        await uploadResourcePdf(selected[0]);
        alert("Upload PDF thành công: 1 file.");
      } else {
        const resp = await uploadResourcePdfs(selected);
        const uploadedCount = Number(resp?.uploaded_count ?? resp?.uploaded?.length ?? 0);
        const errorCount = Number(resp?.error_count ?? resp?.errors?.length ?? 0);
        const topError = resp?.errors?.[0]?.error || "";
        const suffix = errorCount > 0 && topError ? ` (lỗi đầu: ${topError})` : "";
        alert(`Upload PDF xong: thành công ${uploadedCount}, lỗi ${errorCount}.${suffix}`);
      }
      await refreshResources();
    } catch (err) {
      alert(`Lỗi upload resource: ${err.message}`);
    } finally {
      setResourceLoading(false);
      e.target.value = null;
    }
  };

  const handleResourceHtmlUpload = async (e) => {
    const selected = e.target.files ? Array.from(e.target.files) : [];
    if (!selected.length) return;
    setResourceLoading(true);
    try {
      if (selected.length === 1) {
        await uploadResourceHtml(selected[0]);
        alert("Upload HTML thành công: 1 file.");
      } else {
        const resp = await uploadResourceHtmls(selected);
        const uploadedCount = Number(resp?.uploaded_count ?? resp?.uploaded?.length ?? 0);
        const errorCount = Number(resp?.error_count ?? resp?.errors?.length ?? 0);
        const topError = resp?.errors?.[0]?.error || "";
        const suffix = errorCount > 0 && topError ? ` (lỗi đầu: ${topError})` : "";
        alert(`Upload HTML xong: thành công ${uploadedCount}, lỗi ${errorCount}.${suffix}`);
      }
      await refreshResources();
    } catch (err) {
      alert(`Lỗi upload HTML: ${err.message}`);
    } finally {
      setResourceLoading(false);
      e.target.value = null;
    }
  };

  const handleAddUrl = async () => {
    if (!resourceUrl.trim()) return;
    setResourceLoading(true);
    try {
      await addResourceUrl(resourceUrl);
      setResourceUrl("");
      await refreshResources();
    } catch (err) {
      alert(`Lỗi thêm URL: ${err.message}`);
    } finally {
      setResourceLoading(false);
    }
  };

  const handleDeleteResource = async (id) => {
    if (!window.confirm("Bạn có chắc muốn xóa tài nguyên này?")) return;
    setResourceLoading(true);
    try {
      await deleteResource(id);
      await refreshResources();
    } catch (err) {
      alert(`Lỗi xóa: ${err.message}`);
    } finally {
      setResourceLoading(false);
    }
  };

  const handleProgramChange = (sessionId, programId) => {
    setPendingProgramBySession((prev) => ({ ...prev, [sessionId]: programId }));
  };

  const handleConfirmProgram = () => {
    const selected = currentPendingProgramId;
    if (!selected) {
      updateMessages(currentSession, (prev) => [
        ...prev,
        { type: "system", text: "Vui lòng chọn chương trình đào tạo trước khi xác nhận." },
      ]);
      return;
    }
    setSelectedProgramBySession((prev) => ({ ...prev, [currentSession]: selected }));
    setPendingProgramBySession((prev) => ({ ...prev, [currentSession]: selected }));
    const selectedName = programs.find((p) => p.id === selected)?.display_name || selected;
    updateMessages(currentSession, (prev) => [
      ...prev,
      { type: "system", text: `Đã chọn chương trình đào tạo: ${selectedName}` },
    ]);
  };

  const handleSendMessage = async () => {
    if (loading) return;
    if (!inputStr.trim()) return;

    const sessionId = currentSession;
    const selectedProgramId = selectedProgramBySession[sessionId] || "";
    if (!selectedProgramId) {
      updateMessages(sessionId, (prev) => [
        ...prev,
        {
          type: "system",
          text: "Bạn chưa chọn chương trình đào tạo. Vui lòng chọn CTĐT/QH và bấm Xác nhận trước khi gửi câu hỏi.",
        },
      ]);
      return;
    }

    const query = inputStr;

    const transcriptNeedPattern = /(bang diem|tin chi|gpa|lap lich|lich hoc|mon con thieu|thieu mon|hoc ky sau)/i;
    if (!selectedFileIds.length && transcriptNeedPattern.test(query)) {
      updateMessages(sessionId, (prev) => [
        ...prev,
        {
          type: "system",
          text: "Bạn chưa chọn file bảng điểm cho phiên này. Vui lòng tick các file trong mục 'File đã tải lên' rồi gửi lại.",
        },
      ]);
      return;
    }

    setInputStr("");

    if (selectedFileIds.length) {
      const names = files.filter((f) => selectedFileIds.includes(f.file_id)).map((f) => f.file_name).join(", ");
      setProcessingPdf(true);
      setProcessingLabel(names || "Đang xu ly PDF...");
    }

    updateMessages(sessionId, (prev) => [...prev, { type: "user", text: query }]);
    setLoading(true);

    try {
      const response = await askQuestionWithFiles(
        query,
        allowWeb,
        sessionId,
        selectedFileIds,
        selectedProgramId
      );

      if (response?.requires_program_selection) {
        const incomingPrograms = Array.isArray(response?.programs) ? response.programs : [];
        setPrograms(incomingPrograms);
        setPendingProgramBySession((prev) => ({
          ...prev,
          [sessionId]: incomingPrograms[0]?.id || "",
        }));
        setSelectedProgramBySession((prev) => {
          const next = { ...prev };
          delete next[sessionId];
          return next;
        });
        updateMessages(sessionId, (prev) => [
          ...prev,
          {
            type: "system",
            text:
              response?.answer ||
              "Vui lòng chọn lại chương trình đào tạo trước khi tiếp tục.",
          },
        ]);
        return;
      }

      if (response?.selected_program_id) {
        const resolvedProgramId = response.selected_program_id;
        setSelectedProgramBySession((prev) => ({ ...prev, [sessionId]: resolvedProgramId }));
        setPendingProgramBySession((prev) => ({ ...prev, [sessionId]: resolvedProgramId }));
      }

      const answer = typeof response?.answer === "string" ? response.answer : "Không có phản hồi.";
      updateMessages(sessionId, (prev) => [...prev, { type: "bot", text: answer }]);
      const updatedHist = await fetchHistory(sessionId);
      setHistoryList(updatedHist);
    } catch (err) {
      updateMessages(sessionId, (prev) => [...prev, { type: "bot", text: `Loi: ${err.message}` }]);
    } finally {
      setLoading(false);
      setProcessingPdf(false);
      setProcessingLabel("");
    }
  };

  const handleKeyDown = (e) => {
    if (loading) return;
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
                title="Doi ten phien"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRenameSession(s.id);
                }}
                style={{ padding: 6, fontSize: 14 }}
              >
                <i className="fas fa-pen"></i>
              </button>
              <button
                className="icon-btn"
                title="Xoa phien"
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

          {/* Resource Button */}
          <div className="nav-title" style={{ marginTop: 20 }}>Hệ thống</div>
          <div className="history-item" onClick={() => setShowResourcePanel(!showResourcePanel)} style={{ cursor: "pointer", background: showResourcePanel ? "var(--glass-highlight)" : "transparent" }}>
            <i className="fas fa-book"></i> Quản lý Tài nguyên
          </div>

        </div>

        <div className="profile">
          <div className="avatar">U</div>
          <div style={{ fontSize: "14px", fontWeight: "500" }}>User</div>
        </div>
      </aside>

      <main className="main">
        {showResourcePanel && (
          <div className="resource-panel" style={{
            position: "absolute",
            top: 20, right: 20, bottom: 20, width: 350,
            background: "var(--bg-secondary)",
            border: "1px solid var(--glass-border)",
            borderRadius: 12,
            backdropFilter: "blur(20px)",
            padding: 20,
            zIndex: 100,
            display: "flex", flexDirection: "column",
            boxShadow: "0 4px 30px rgba(0,0,0,0.5)"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 15 }}>
              <h3 style={{ margin: 0 }}>Tài nguyên RAG</h3>
              <button className="icon-btn" onClick={() => setShowResourcePanel(false)}><i className="fas fa-times"></i></button>
            </div>

            <div style={{ flex: 1, overflowY: "auto", marginBottom: 15 }}>
              {resourceLoading && <div style={{ textAlign: "center", color: "#94a3b8" }}><i className="fas fa-circle-notch fa-spin"></i> Loading...</div>}
              {!resourceLoading && resources.map((res, i) => (
                <div key={i} style={{
                  padding: "8px 10px",
                  background: "rgba(255,255,255,0.05)",
                  borderRadius: 6,
                  marginBottom: 6,
                  fontSize: "13px",
                  display: "flex",
                  alignItems: "center"
                }}>
                  <i className={`fas ${res.type === 'url' ? 'fa-globe' : 'fa-file-pdf'}`} style={{ marginRight: 8, color: res.type === 'url' ? '#60a5fa' : '#f87171' }}></i>
                  <span style={{ wordBreak: "break-all", flex: 1 }}>{res.name}</span>
                  <button onClick={() => handleDeleteResource(res.id)} style={{ background: "transparent", border: "none", color: "#94a3b8", cursor: "pointer", padding: "0 5px" }}>
                    <i className="fas fa-trash-alt"></i>
                  </button>
                </div>
              ))}
              {!resourceLoading && resources.length === 0 && <div style={{ color: "#64748b", fontSize: 13 }}>Chưa có tài nguyên nào.</div>}
            </div>

            <div style={{ borderTop: "1px solid var(--glass-border)", paddingTop: 15 }}>
              <div style={{ marginBottom: 10, fontSize: 13, fontWeight: "bold" }}>Thêm PDF Sổ tay</div>
              <input type="file" ref={resourceFileInputRef} accept="application/pdf" multiple style={{ display: "none" }} onChange={handleResourceUpload} />
              <button className="chip-btn" onClick={() => resourceFileInputRef.current?.click()} style={{ width: "100%", justifyContent: "center" }}>
                <i className="fas fa-upload"></i> Upload PDF
              </button>
              <div style={{ marginTop: 6, color: "#64748b", fontSize: 12 }}>Bạn có thể chọn nhiều file PDF cùng lúc.</div>

              <div style={{ marginBottom: 10, marginTop: 15, fontSize: 13, fontWeight: "bold" }}>Thêm HTML Local</div>
              <input type="file" ref={resourceHtmlInputRef} accept=".html,.htm" multiple style={{ display: "none" }} onChange={handleResourceHtmlUpload} />
              <button className="chip-btn" onClick={() => resourceHtmlInputRef.current?.click()} style={{ width: "100%", justifyContent: "center" }}>
                <i className="fas fa-code"></i> Upload HTML
              </button>
              <div style={{ marginTop: 6, color: "#64748b", fontSize: 12 }}>Bạn có thể chọn nhiều file HTML cùng lúc.</div>

              <div style={{ marginTop: 15, marginBottom: 10, fontSize: 13, fontWeight: "bold" }}>Thêm Link Quy chế</div>
              <div style={{ display: "flex", gap: 5 }}>
                <input
                  value={resourceUrl}
                  onChange={(e) => setResourceUrl(e.target.value)}
                  placeholder="https://uet.edu.vn/..."
                  style={{
                    flex: 1,
                    background: "rgba(0,0,0,0.2)",
                    border: "1px solid var(--glass-border)",
                    borderRadius: 6,
                    padding: "6px 10px",
                    color: "white",
                    fontSize: 13
                  }}
                />
                <button className="chip-btn" onClick={handleAddUrl} disabled={!resourceUrl}>
                  <i className="fas fa-plus"></i>
                </button>
              </div>
            </div>
          </div>
        )}

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
                    <div className="msg-content bot-text">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                    </div>
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
                <i className="fas fa-circle-notch fa-spin"></i> Ðang suy nghi...
              </div>
            </div>
          )}
          <div ref={chatEndRef}></div>
        </div>

        <div className="input-region">
          <div className="input-container">
            <div className={`program-selector ${currentSelectedProgramId ? "has-selection" : "required-selection"}`}>
              <div className="program-selector-header">
                <div className="program-selector-title">
                  <i className="fas fa-graduation-cap"></i>
                  <span>
                    {currentSelectedProgramId
                      ? "Chương trình đào tạo hiện tại"
                      : "Chọn chương trình đào tạo/QH (bắt buộc)"}
                  </span>
                </div>
                <button
                  className="chip-btn"
                  onClick={() => refreshPrograms(true)}
                  disabled={programsLoading}
                  title="Làm mới danh sách chương trình"
                >
                  <i className={`fas ${programsLoading ? "fa-circle-notch fa-spin" : "fa-sync-alt"}`}></i>
                </button>
              </div>
              <div className="program-selector-controls">
                <select
                  value={currentPendingProgramId}
                  onChange={(e) => handleProgramChange(currentSession, e.target.value)}
                  disabled={programsLoading || !programs.length}
                >
                  {!programs.length && (
                    <option value="">
                      {programsLoading ? "Đang tải chương trình..." : "Không có chương trình khả dụng"}
                    </option>
                  )}
                  {groupedPrograms.map((group) => (
                    <optgroup key={group.groupName} label={group.groupName}>
                      {group.items.map((program) => (
                        <option key={program.id} value={program.id}>
                          {program.qh_label || program.display_name || program.name || program.id}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <button
                  className="chip-btn program-confirm-btn"
                  onClick={handleConfirmProgram}
                  disabled={programsLoading || !currentPendingProgramId}
                >
                  Xác nhận
                </button>
              </div>
              {currentProgramDisplayName ? (
                <div className="program-selector-hint">Đang dùng: {currentProgramDisplayName}</div>
              ) : (
                <div className="program-selector-hint">
                  Hệ thống chỉ xử lý câu hỏi sau khi bạn xác nhận CTĐT/QH.
                </div>
              )}
            </div>

            {processingPdf && (
              <div className="processing-banner">
                <i className="fas fa-circle-notch fa-spin"></i> {processingLabel || "Đang xử lý PDF..."}
              </div>
            )}
            {visibleFiles.length > 0 && (
              <div className="file-gallery">
                <div className="file-gallery-header">
                  <div className="file-gallery-title">
                    <i className="fas fa-folder-open"></i> File đã tải lên
                  </div>
                  <div className="file-gallery-actions">
                    <button className="chip-btn" onClick={refreshFiles} title="Làm mới danh sách">
                      <i className="fas fa-sync-alt"></i>
                    </button>
                    <button className="chip-btn" onClick={handleSelectAllFiles} title="Chọn tất cả">
                      <i className="fas fa-check-double"></i>
                    </button>
                  </div>
                </div>
                <div className="file-chip-list">
                  {visibleFiles.map((f) => {
                    const active = selectedFileIds.includes(f.file_id);
                    return (
                      <div
                        key={f.file_id}
                        className={`file-chip ${active ? "selected" : ""}`}
                        onClick={() => handleToggleFile(f.file_id)}
                        title={f.file_name}
                      >
                        <div className="file-chip-icon">
                          <i className="fas fa-file-pdf"></i>
                        </div>
                        <div className="file-chip-body">
                          <div className="file-chip-name">{f.file_name}</div>
                          <div className="file-chip-meta">PDF</div>
                        </div>
                        <div className="file-chip-check">
                          {active ? <i className="fas fa-check-circle"></i> : <i className="far fa-circle"></i>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Chi hien chip preview cu neu chua co gallery (khong co danh sach files tu server) */}
            {!files.length && (selectedNames.length || uploadedFile) && (
              <div className="file-preview">
                <i className="fas fa-file-pdf"></i> {selectedNames.length ? selectedNames.join(", ") : uploadedFile}
                <i
                  className="fas fa-times"
                  style={{ cursor: "pointer", marginLeft: 5 }}
                  onClick={() => {
                    setUploadedFile(null);
                    updateSelectedFiles(currentSession, []);
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
              <button className="icon-btn" title="Tai len PDF" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
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
                disabled={loading}
              ></textarea>

              <button
                className="icon-btn"
                style={{ color: inputStr && currentSelectedProgramId ? "#3b82f6" : "inherit" }}
                onClick={handleSendMessage}
                disabled={loading || !inputStr.trim() || !currentSelectedProgramId}
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
