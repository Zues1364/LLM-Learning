
import { useEffect, useRef, useState, useCallback, useMemo, isValidElement } from "react";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./style.css";

const DEFAULT_API_BASE = "http://127.0.0.1:9000";
const ENV_API_BASE =
  typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE
    ? String(import.meta.env.VITE_API_BASE).trim()
    : "";
const inferApiBase = () => {
  if (typeof window === "undefined") return DEFAULT_API_BASE;
  const protocol = window.location.protocol || "http:";
  const host = window.location.hostname || "127.0.0.1";
  return `${protocol}//${host}:9000`;
};
const API_BASE = ENV_API_BASE || inferApiBase();
const APP_AUTH_CALLBACK_URI = `${API_BASE}/api/auth/google/callback`;
const MAIL_CONNECT_CALLBACK_URI = `${API_BASE}/api/mail/connect/callback`;
const createSessionId = () =>
  (crypto?.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`);

const readApiErrorMessage = async (res) => {
  const text = await res.text();
  if (!text) return res.statusText || "Request failed";
  try {
    const data = JSON.parse(text);
    if (typeof data?.detail === "string") return data.detail;
    if (data?.detail) return JSON.stringify(data.detail);
    if (typeof data?.message === "string") return data.message;
  } catch {
    // Keep the original response text when it is not JSON.
  }
  return text;
};

const normalizeQueryForIntent = (text) =>
  String(text || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();

const stripSourceFooterFromMessage = (value) => {
  const text = String(value || "");
  if (!text.trim()) return text;

  const normalizeLine = (line) =>
    normalizeQueryForIntent(line)
      .replace(/^[>\s\-\*\+`#]+/g, "")
      .replace(/[^a-z0-9.\-:\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();

  const isSourceHeader = (line) => {
    const norm = normalizeLine(line);
    if (!norm) return false;
    const tokens = norm.split(" ").filter(Boolean);
    const first = tokens[0] || "";
    const hasNguonLead = first.startsWith("ngu");
    const hasTham = tokens.some((token) => token.startsWith("tham"));
    const hasChieu = tokens.some((token) => token.startsWith("chi"));
    const hasKhao = tokens.some((token) => token.startsWith("kha"));
    if (hasNguonLead && ((hasTham && hasChieu) || hasKhao)) return true;
    if (norm === "nguon" || norm === "nguon:" || norm.startsWith("nguon tham chieu") || norm.startsWith("nguon tham khao")) {
      return true;
    }
    return false;
  };

  const looksLikeSourceItem = (line) => {
    const raw = String(line || "").trim();
    if (!raw) return true;
    if (/^(?:[-*]\s*)?\[\d+\]\s+.+$/.test(raw)) return true;
    const norm = normalizeLine(raw);
    return ["pdf", "html", "sheet", "page", "line", "chunk", ".xlsx", ".doc", ".ppt"].some((token) =>
      norm.includes(token)
    );
  };

  const trimRightBlankLines = (arr) => {
    const cloned = [...arr];
    while (cloned.length && !String(cloned[cloned.length - 1] || "").trim()) cloned.pop();
    return cloned;
  };

  const lines = text.split("\n");
  let end = lines.length - 1;
  while (end >= 0 && !String(lines[end] || "").trim()) end -= 1;
  if (end < 0) return "";

  const tailNorm = normalizeLine(lines[end]);
  if (tailNorm.startsWith("nguon:") && looksLikeSourceItem(lines[end])) {
    const kept = lines.slice(0, end);
    while (kept.length && !String(kept[kept.length - 1] || "").trim()) kept.pop();
    return kept.join("\n");
  }

  let headerIdx = -1;
  for (let i = end; i >= 0; i -= 1) {
    if (isSourceHeader(lines[i])) {
      headerIdx = i;
      break;
    }
  }
  if (headerIdx < 0) {
    // Fallback: remove trailing [n] source lines, even if header is OCR-broken.
    let tail = end;
    let tailSourceCount = 0;
    while (tail >= 0) {
      const raw = String(lines[tail] || "").trim();
      if (!raw) {
        tail -= 1;
        continue;
      }
      if (!looksLikeSourceItem(raw)) break;
      tailSourceCount += 1;
      tail -= 1;
    }
    if (tailSourceCount >= 2) {
      const beforeTailNorm = tail >= 0 ? normalizeLine(lines[tail]) : "";
      if (
        !beforeTailNorm ||
        beforeTailNorm.startsWith("ngu") ||
        beforeTailNorm.includes("tham chi") ||
        beforeTailNorm.includes("tham kha")
      ) {
        return trimRightBlankLines(lines.slice(0, Math.max(0, tail))).join("\n");
      }
    }
    return text;
  }

  const trailing = lines.slice(headerIdx + 1, end + 1).filter((line) => String(line || "").trim());
  if (!trailing.length) return lines.slice(0, headerIdx).join("\n").trimEnd();

  const sourceLikeCount = trailing.filter((line) => looksLikeSourceItem(line)).length;
  if (sourceLikeCount >= Math.max(1, Math.floor(trailing.length * 0.6))) {
    return lines.slice(0, headerIdx).join("\n").trimEnd();
  }

  return text;
};

const citationDisplayLabel = (citation, fallbackIndex) => {
  const sourceFile = String(citation?.source_file || "").trim();
  const chunk = Number.isInteger(citation?.chunk_index) ? citation.chunk_index : null;
  const page = Number.isInteger(citation?.page) ? citation.page : null;
  const idx = Number.isInteger(citation?.id) ? citation.id : fallbackIndex + 1;
  const locationParts = [];
  if (page !== null) locationParts.push(`Page ${page}`);
  if (chunk !== null) locationParts.push(`Chunk ${chunk}`);
  if (sourceFile && locationParts.length > 0) return `[${idx}] ${sourceFile} - ${locationParts.join(" - ")}`;
  if (locationParts.length > 0) return `[${idx}] ${locationParts.join(" - ")}`;
  if (sourceFile) return `[${idx}] ${sourceFile}`;
  return `[${idx}] Nguồn ${idx}`;
};

const citationAnchorLabel = (citation, fallbackIndex) => {
  const idx = Number.isInteger(citation?.id) ? citation.id : fallbackIndex + 1;
  const page = Number.isInteger(citation?.page) ? citation.page : null;
  const chunk = Number.isInteger(citation?.chunk_index) ? citation.chunk_index : null;
  if (page !== null) return `[${idx}] Tr.${page}`;
  if (chunk !== null) return `[${idx}] Chunk ${chunk}`;

  const excerpt = String(citation?.excerpt || "");
  const classMatch = excerpt.match(/\b([A-Z]{2,4}\d{3,4}[A-Z]?\s*\d{1,3})\b/);
  if (classMatch) return `[${idx}] ${classMatch[1].replace(/\s+/g, " ").trim()}`;

  return `[${idx}] nguồn`;
};

const normalizeCitationMatchText = (value) =>
  normalizeQueryForIntent(value)
    .replace(/[^a-z0-9.\-\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const CITATION_STOPWORDS = new Set([
  "la", "va", "cua", "cho", "voi", "trong", "theo", "nay", "kia", "do", "den", "tu", "mot",
  "cac", "nhung", "duoc", "se", "da", "co", "khong", "neu", "thi", "ban", "toi", "chung",
  "em", "anh", "chi", "day", "rang", "nhu", "de", "o", "tai", "ve", "hay", "roi", "van",
  "coi", "nua", "luon", "them", "rat", "qua", "tren", "duoi", "sau", "truoc", "giua",
  "the", "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "is", "are", "was",
  "were", "be", "been", "being", "with", "by", "as", "that", "this", "these", "those",
]);

const tokenizeCitationMatchText = (value) =>
  normalizeCitationMatchText(value)
    .split(" ")
    .filter((token) => token.length >= 2)
    .filter((token) => !CITATION_STOPWORDS.has(token))
    .filter((token) => !/^\d$/.test(token));

const FORCE_CITATION_KEYWORDS = [
  "ielts",
  "toeic",
  "toefl",
  "vstep",
  "aptis",
  "jlpt",
  "nat-test",
  "j-test",
];

const extractNodePlainText = (node) => {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map((item) => extractNodePlainText(item)).join(" ");
  if (isValidElement(node)) return extractNodePlainText(node.props?.children);
  return "";
};

const matchLineCitations = (lineText, citations, maxMatches = 2) => {
  const line = String(lineText || "").trim();
  if (!line || !Array.isArray(citations) || !citations.length) return [];

  const lineNorm = normalizeCitationMatchText(line);
  const lineTokens = tokenizeCitationMatchText(line);
  if (!lineTokens.length) return [];
  const lineTokenSet = new Set(lineTokens);
  const lineUpper = line.toUpperCase();
  const lineCodes = lineUpper.match(/\b[A-Z]{2,4}\d{3,4}[A-Z]?\b/g) || [];
  const forceKeywords = FORCE_CITATION_KEYWORDS.filter((key) => lineNorm.includes(key));
  const hasStrongSignal =
    lineCodes.length > 0 ||
    /\b(?:ielts|toeic|toefl|gpa|tin\s*chi|phong|ca|thu|lop|gv|giang\s*v(i|ie)n)\b/i.test(line) ||
    /\d/.test(line);

  const scored = [];
  for (let index = 0; index < citations.length; index += 1) {
    const citation = citations[index] || {};
    const excerpt = String(citation.excerpt || "");
    const source = String(citation.source_file || "");
    const haystack = `${excerpt} ${source}`.trim();
    const hayNorm = normalizeCitationMatchText(haystack);
    if (forceKeywords.length > 0 && !forceKeywords.some((key) => hayNorm.includes(key))) {
      continue;
    }
    const hayTokens = tokenizeCitationMatchText(haystack);
    if (!hayTokens.length) continue;
    const hayTokenSet = new Set(hayTokens);

    let score = 0;
    for (const token of lineTokenSet) {
      if (hayTokenSet.has(token)) score += 1;
    }

    const citationUpper = haystack.toUpperCase();
    if (lineCodes.length) {
      for (const code of lineCodes) {
        if (citationUpper.includes(code)) score += 4;
      }
    }

    if (score <= 0) continue;
    scored.push({ citation, score, index });
  }

  scored.sort((a, b) => (b.score - a.score) || (a.index - b.index));
  const topScore = scored[0]?.score || 0;
  const dynamicFloor = topScore >= 5 ? topScore - 2 : topScore >= 3 ? topScore - 1 : topScore;
  const minScore = hasStrongSignal ? 2 : 3;
  const threshold = Math.max(minScore, dynamicFloor);
  const filtered = scored.filter((item) => item.score >= threshold);
  return filtered.slice(0, maxMatches).map((item) => item.citation);
};

const shouldSkipCitationLine = (lineText) => {
  const line = String(lineText || "").trim();
  if (!line) return true;
  const norm = normalizeCitationMatchText(line);
  if (!norm) return true;

  const hasCourseCode = /\b[A-Z]{2,4}\d{3,4}[A-Z]?\b/.test(String(line).toUpperCase());
  const hasDigit = /\d/.test(line);
  const endsWithColon = /[:：]\s*$/.test(line);
  const sectionHeading = /^\s*\d+\)\s*[^\d:]+:?$/i.test(line);
  const courseTitleHeading = /^\s*(?:[-*]\s*)?[A-Z]{2,4}\d{3,4}[A-Z]?\s*-\s*[^:]+$/i.test(line);
  const hasScheduleDetailSignal = /\b(thu|ca|phong|gv|giang vien|lop)\b/.test(norm);

  if (endsWithColon && !hasCourseCode && !hasDigit) return true;
  if (sectionHeading && !hasCourseCode) return true;
  if (courseTitleHeading && !hasScheduleDetailSignal) return true;

  const headingPhrases = [
    "chao ban",
    "duoi day",
    "lich hoc theo tung mon",
    "cac lop cua giang vien",
    "goi y lich",
    "goi y",
    "luu y",
    "thieu tin chi",
    "mon con thieu uu tien",
    "gpa projection",
    "nguon tham chieu",
    "nguon tham khao",
    "nguon",
  ];
  if (headingPhrases.some((phrase) => norm === phrase || norm.startsWith(`${phrase} `))) {
    return true;
  }

  return false;
};

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
    credentials: "include",
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
  const res = await fetch(`${API_BASE}/history?session_id=${encodeURIComponent(sessionId || "")}`, {
    credentials: "include",
  });
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
    credentials: "include",
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const normalizeChatSession = (item, index = 0) => {
  const id = String(item?.id || item?.session_id || "").trim();
  if (!id) return null;
  return {
    id,
    title: String(item?.title || `Phien ${index + 1}`).trim() || `Phien ${index + 1}`,
    selected_program_id: item?.selected_program_id || "",
    selected_file_ids: Array.isArray(item?.selected_file_ids) ? item.selected_file_ids.filter(Boolean) : [],
  };
};

const normalizeChatMessage = (item) => {
  const role = String(item?.role || "").toLowerCase();
  const type = role === "user" ? "user" : role === "system" ? "system" : "bot";
  return {
    type,
    text: String(item?.content || ""),
    citations: Array.isArray(item?.citations) ? item.citations : [],
  };
};

async function fetchChatSessions() {
  const res = await fetch(`${API_BASE}/api/chat/sessions`, { credentials: "include" });
  if (!res.ok) throw new Error(await readApiErrorMessage(res));
  return res.json();
}

async function createChatSessionApi(sessionId, title, selectedProgramId = "", selectedFileIds = []) {
  const res = await fetch(`${API_BASE}/api/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      session_id: sessionId,
      title,
      selected_program_id: selectedProgramId || null,
      selected_file_ids: selectedFileIds || [],
    }),
  });
  if (!res.ok) throw new Error(await readApiErrorMessage(res));
  return res.json();
}

async function fetchChatMessages(sessionId) {
  const res = await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`, {
    credentials: "include",
  });
  if (!res.ok) throw new Error(await readApiErrorMessage(res));
  return res.json();
}

async function updateChatSessionApi(sessionId, payload) {
  const res = await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) throw new Error(await readApiErrorMessage(res));
  return res.json();
}

async function archiveChatSessionApi(sessionId) {
  const res = await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await readApiErrorMessage(res));
  return res.json();
}

// --- Resource API ---
async function fetchResources(sessionId) {
  const url = new URL(`${API_BASE}/api/resources`);
  if (sessionId) url.searchParams.set("session_id", sessionId);
  const res = await fetch(url.toString(), { credentials: "include" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchPrograms(refresh = false) {
  const url = refresh ? `${API_BASE}/api/programs?refresh=true` : `${API_BASE}/api/programs`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadResourceFiles(files, sessionId) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  if (sessionId) form.append("session_id", sessionId);
  const res = await fetch(`${API_BASE}/api/resources/upload`, { method: "POST", credentials: "include", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function addResourceUrl(url, sessionId) {
  const res = await fetch(`${API_BASE}/api/resources/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ url, session_id: sessionId || null }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function deleteResource(id, sessionId) {
  const url = new URL(`${API_BASE}/api/resources/${encodeURIComponent(id)}`);
  if (sessionId) url.searchParams.set("session_id", sessionId);
  const res = await fetch(url.toString(), { method: "DELETE", credentials: "include" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchMailStatus(sessionId) {
  const url = new URL(`${API_BASE}/api/mail/status`);
  url.searchParams.set("session_id", sessionId || "");
  const res = await fetch(url.toString(), { credentials: "include" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchAuthMe() {
  const res = await fetch(`${API_BASE}/api/auth/me`, { credentials: "include" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function startGoogleSignIn(sessionId, redirectUri) {
  const res = await fetch(`${API_BASE}/api/auth/google/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      session_id: sessionId || null,
      redirect_uri: redirectUri || null,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function logoutGoogleSignIn() {
  const res = await fetch(`${API_BASE}/api/auth/logout`, {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function startMailConnect(sessionId, redirectUri) {
  const res = await fetch(`${API_BASE}/api/mail/connect/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      session_id: sessionId,
      redirect_uri: redirectUri || null,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function disconnectMail(sessionId) {
  const res = await fetch(`${API_BASE}/api/mail/disconnect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchMailWhitelist(sessionId) {
  const url = new URL(`${API_BASE}/api/mail/whitelist`);
  url.searchParams.set("session_id", sessionId || "");
  const res = await fetch(url.toString(), { credentials: "include" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function setMailWhitelist(sessionId, senders) {
  const res = await fetch(`${API_BASE}/api/mail/whitelist`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ session_id: sessionId, senders }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function pollMailNow(sessionId) {
  const res = await fetch(`${API_BASE}/api/mail/poll`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await readApiErrorMessage(res));
  return res.json();
}

async function fetchMailCandidates(sessionId, status = "") {
  const url = new URL(`${API_BASE}/api/mail/candidates`);
  url.searchParams.set("session_id", sessionId || "");
  if (status) url.searchParams.set("status", status);
  const res = await fetch(url.toString(), { credentials: "include" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function applyMailCandidate(sessionId, candidateId) {
  const res = await fetch(`${API_BASE}/api/mail/candidates/${encodeURIComponent(candidateId)}/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function rejectMailCandidate(sessionId, candidateId, reason = "") {
  const res = await fetch(`${API_BASE}/api/mail/candidates/${encodeURIComponent(candidateId)}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ session_id: sessionId, reason }),
  });
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
  const [loadingFrame, setLoadingFrame] = useState(0);
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
  const [resourceSearch, setResourceSearch] = useState("");
  const [resourceListVisible, setResourceListVisible] = useState(false);
  const [resourceListExpanded, setResourceListExpanded] = useState(false);
  const [mailStatus, setMailStatus] = useState(null);
  const [mailCandidates, setMailCandidates] = useState([]);
  const [mailWhitelistText, setMailWhitelistText] = useState("");
  const [mailLoading, setMailLoading] = useState(false);
  const [mailError, setMailError] = useState("");
  const [expandedMailCandidates, setExpandedMailCandidates] = useState({});
  const [authState, setAuthState] = useState({
    loading: false,
    authenticated: false,
    email: "",
    userId: "",
    name: "",
  });
  const [mailMode, setMailMode] = useState("session");
  const [programs, setPrograms] = useState([]);
  const [programsLoading, setProgramsLoading] = useState(false);
  const [citationViewer, setCitationViewer] = useState(null);
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
  const resourceUploadInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const filesRef = useRef([]);
  const authPopupPollRef = useRef(null);

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
  const mailConnected = Boolean(mailStatus?.connected);
  const isGoogleSignedIn = Boolean(authState?.authenticated);
  const canManageMail = isGoogleSignedIn;
  const mailPendingCount = Number(mailStatus?.candidate_counts?.pending || 0);
  const mailStatusText = useMemo(() => {
    const modeLabel = mailMode === "user" ? "User" : "Session";
    if (mailConnected) {
      const connectedEmail = mailStatus?.email || authState?.email || "unknown";
      return `Mode: ${modeLabel} | Connected: ${connectedEmail} | Pending: ${mailPendingCount}`;
    }
    if (authState?.authenticated) {
      const signedInEmail = authState?.email || "unknown";
      return `Mode: ${modeLabel} | Signed in: ${signedInEmail} | Gmail chưa kết nối`;
    }
    return `Mode: ${modeLabel} | Chưa đăng nhập Google`;
  }, [authState, mailConnected, mailMode, mailPendingCount, mailStatus]);
  const filteredResources = useMemo(() => {
    const keyword = resourceSearch.trim().toLowerCase();
    if (!keyword) return resources;
    return resources.filter((res) => String(res?.name || "").toLowerCase().includes(keyword));
  }, [resources, resourceSearch]);
  const resourcePreviewLimit = 8;
  const visibleResources = resourceListExpanded
    ? filteredResources
    : filteredResources.slice(0, resourcePreviewLimit);
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

  const refreshResources = useCallback(async (sessionId = currentSession) => {
    try {
      const data = await fetchResources(sessionId);
      setResources(data);
    } catch (err) {
      console.error("Fetch resources failed", err);
    }
  }, [currentSession]);

  const refreshMailState = useCallback(async (sessionId = currentSession) => {
    if (!sessionId) return;
    try {
      setMailError("");
      setAuthState((prev) => ({ ...prev, loading: true }));
      const [authData, statusData, whitelistData, candidateData] = await Promise.all([
        fetchAuthMe().catch(() => ({ authenticated: false })),
        fetchMailStatus(sessionId),
        fetchMailWhitelist(sessionId),
        fetchMailCandidates(sessionId, "pending"),
      ]);
      setAuthState({
        loading: false,
        authenticated: Boolean(authData?.authenticated),
        email: authData?.user?.email || "",
        userId: authData?.user?.id || "",
        name: authData?.user?.name || "",
      });
      setMailStatus(statusData);
      setMailWhitelistText((whitelistData?.senders || []).join("\n"));
      setMailCandidates(candidateData?.candidates || []);
      setMailMode(Boolean(authData?.authenticated) ? "user" : "session");
    } catch (err) {
      console.error("Fetch mail state failed", err);
      setMailError(err.message || "Không tải được trạng thái mail.");
      setAuthState((prev) => ({ ...prev, loading: false }));
    }
  }, [currentSession]);

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

  const refreshChatSessions = useCallback(async () => {
    if (!authState.authenticated) return;
    try {
      const data = await fetchChatSessions();
      const serverSessions = (data?.sessions || []).map(normalizeChatSession).filter(Boolean);
      if (!serverSessions.length) return;

      setSessions(serverSessions.map(({ id, title }) => ({ id, title })));
      setSelectedProgramBySession((prev) => {
        const next = { ...prev };
        serverSessions.forEach((session) => {
          if (session.selected_program_id) next[session.id] = session.selected_program_id;
        });
        return next;
      });
      setPendingProgramBySession((prev) => {
        const next = { ...prev };
        serverSessions.forEach((session) => {
          if (session.selected_program_id) next[session.id] = session.selected_program_id;
        });
        return next;
      });
      setSelectedFilesBySession((prev) => {
        const next = { ...prev };
        serverSessions.forEach((session) => {
          if (session.selected_file_ids.length) next[session.id] = session.selected_file_ids;
        });
        return next;
      });
      if (!serverSessions.some((session) => session.id === currentSession)) {
        setCurrentSession(serverSessions[0].id);
      }
    } catch (err) {
      if (!String(err?.message || "").includes("401")) {
        console.error("Fetch chat sessions failed", err);
      }
    }
  }, [authState.authenticated, currentSession]);

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
    refreshChatSessions();
  }, [authState.authenticated, authState.userId, refreshChatSessions]);

  useEffect(() => {
    if (!authState.authenticated || !currentSession) return;
    fetchChatMessages(currentSession)
      .then((data) => {
        const serverMessages = (data?.messages || []).map(normalizeChatMessage).filter((msg) => msg.text);
        if (serverMessages.length) {
          setMessagesBySession((prev) => ({ ...prev, [currentSession]: serverMessages }));
        }
      })
      .catch((err) => {
        if (!String(err?.message || "").includes("401") && !String(err?.message || "").includes("404")) {
          console.error("Fetch chat messages failed", err);
        }
      });
  }, [authState.authenticated, authState.userId, currentSession]);

  useEffect(() => {
    refreshFiles();
    refreshResources(currentSession);
    refreshMailState(currentSession);
    refreshPrograms(false);
  }, [currentSession, refreshFiles, refreshResources, refreshMailState, refreshPrograms]);

  useEffect(() => {
    const handleWindowFocus = () => {
      refreshMailState(currentSession);
      refreshResources(currentSession);
    };
    window.addEventListener("focus", handleWindowFocus);
    return () => window.removeEventListener("focus", handleWindowFocus);
  }, [currentSession, refreshMailState, refreshResources]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentMessages, loading]);

  useEffect(() => {
    if (!loading) {
      setLoadingFrame(0);
      return;
    }
    const timer = window.setInterval(() => {
      setLoadingFrame((prev) => (prev + 1) % 3);
    }, 280);
    return () => window.clearInterval(timer);
  }, [loading]);

  const clearAuthPopupPoll = useCallback(() => {
    if (authPopupPollRef.current) {
      window.clearInterval(authPopupPollRef.current);
      authPopupPollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => clearAuthPopupPoll();
  }, [clearAuthPopupPoll]);

  const openOAuthPopupAndRefresh = useCallback(
    (authUrl, waitFor = "auth") => {
      if (!authUrl) throw new Error("Thiếu URL xác thực OAuth.");

      clearAuthPopupPoll();
      const popup = window.open(authUrl, "google-oauth", "width=540,height=760");
      if (!popup) {
        window.open(authUrl, "_blank", "noopener,noreferrer");
        setTimeout(() => {
          refreshMailState(currentSession);
          refreshResources(currentSession);
        }, 2000);
        return;
      }

      const startedAt = Date.now();
      authPopupPollRef.current = window.setInterval(() => {
        const timedOut = Date.now() - startedAt > 5 * 60 * 1000;
        const waitForMail = waitFor === "mail";
        const statusPromise = waitForMail
          ? Promise.all([fetchAuthMe(), fetchMailStatus(currentSession)]).then(([authData, mailData]) => ({
              authData,
              mailData,
            }))
          : fetchAuthMe().then((authData) => ({ authData, mailData: null }));

        statusPromise
          .then(async ({ authData, mailData }) => {
            const completed = waitForMail ? Boolean(mailData?.connected) : Boolean(authData?.authenticated);
            if (completed) {
              clearAuthPopupPoll();
              try {
                if (!popup.closed) popup.close();
              } catch {
                /* ignore */
              }
              await refreshMailState(currentSession);
              await refreshResources(currentSession);
              return;
            }

            if (!popup.closed && !timedOut) return;
            clearAuthPopupPoll();
            await refreshMailState(currentSession);
            await refreshResources(currentSession);
          })
          .catch(async (err) => {
            if (!popup.closed && !timedOut) return;
            clearAuthPopupPoll();
            try {
              await refreshMailState(currentSession);
              await refreshResources(currentSession);
            } catch (refreshErr) {
              console.error("Refresh after OAuth popup failed", refreshErr);
            }
            console.error("OAuth popup polling failed", err);
          });
      }, 1000);
    },
    [clearAuthPopupPoll, currentSession, refreshMailState, refreshResources]
  );

  const handleNewChat = async () => {
    let newId = createSessionId();
    let newTitle = `Phien ${sessions.length + 1}`;
    if (authState.authenticated) {
      try {
        const created = await createChatSessionApi(newId, newTitle, programs[0]?.id || "", []);
        const serverSession = normalizeChatSession(created?.session, sessions.length);
        if (serverSession) {
          newId = serverSession.id;
          newTitle = serverSession.title;
        }
      } catch (err) {
        console.error("Create server chat session failed", err);
      }
    }
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
      if (authState.authenticated) {
        await archiveChatSessionApi(sessionId);
      }
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

  const handleRenameSession = async (sessionId) => {
    const target = sessions.find((s) => s.id === sessionId);
    if (!target) return;
    const nextTitle = window.prompt("Nhap ten phien", target.title)?.trim();
    if (!nextTitle) return;
    if (authState.authenticated) {
      try {
        await updateChatSessionApi(sessionId, { title: nextTitle });
      } catch (err) {
        console.error("Rename server chat session failed", err);
      }
    }
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

  const handleResourceFilesUpload = async (e) => {
    const selected = e.target.files ? Array.from(e.target.files) : [];
    if (!selected.length) return;
    const sessionId = currentSession;
    setResourceLoading(true);
    try {
      const resp = await uploadResourceFiles(selected, sessionId);
      const uploadedCount = Number(resp?.uploaded_count ?? resp?.uploaded?.length ?? 0);
      const errorCount = Number(resp?.error_count ?? resp?.errors?.length ?? 0);
      const uploadedPdfCount = Number(resp?.uploaded_pdf_count ?? 0);
      const uploadedHtmlCount = Number(resp?.uploaded_html_count ?? 0);
      const topError = resp?.errors?.[0]?.error || "";
      const suffix = errorCount > 0 && topError ? ` (lỗi đầu: ${topError})` : "";
      alert(
        `Upload resource xong: thành công ${uploadedCount} (PDF: ${uploadedPdfCount}, HTML: ${uploadedHtmlCount}), lỗi ${errorCount}.${suffix}`
      );
      await refreshResources(sessionId);
    } catch (err) {
      alert(`Lỗi upload resource: ${err.message}`);
    } finally {
      setResourceLoading(false);
      e.target.value = null;
    }
  };

  const handleAddUrl = async () => {
    if (!resourceUrl.trim()) return;
    const sessionId = currentSession;
    setResourceLoading(true);
    try {
      await addResourceUrl(resourceUrl, sessionId);
      setResourceUrl("");
      await refreshResources(sessionId);
    } catch (err) {
      alert(`Lỗi thêm URL: ${err.message}`);
    } finally {
      setResourceLoading(false);
    }
  };

  const handleDeleteResource = async (id) => {
    if (!window.confirm("Bạn có chắc muốn xóa tài nguyên này?")) return;
    const sessionId = currentSession;
    setResourceLoading(true);
    try {
      await deleteResource(id, sessionId);
      await refreshResources(sessionId);
    } catch (err) {
      alert(`Lỗi xóa: ${err.message}`);
    } finally {
      setResourceLoading(false);
    }
  };

  const handleMailConnect = async () => {
    const sessionId = currentSession;
    if (!sessionId) return;
    if (!canManageMail) {
      const msg = "Vui lòng đăng nhập Google ở khu vực User trước khi kết nối Gmail.";
      setMailError(msg);
      alert(msg);
      return;
    }
    setMailLoading(true);
    setMailError("");
    try {
      const data = await startMailConnect(sessionId, MAIL_CONNECT_CALLBACK_URI);
      if (data?.auth_url) {
        openOAuthPopupAndRefresh(data.auth_url, "mail");
      } else {
        alert("Không tạo được URL OAuth.");
      }
    } catch (err) {
      alert(`Lỗi kết nối Gmail: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    const sessionId = currentSession;
    if (!sessionId) return;
    setMailLoading(true);
    setMailError("");
    try {
      const data = await startGoogleSignIn(sessionId, APP_AUTH_CALLBACK_URI);
      if (data?.auth_url) {
        openOAuthPopupAndRefresh(data.auth_url);
      } else {
        throw new Error("Không tạo được URL đăng nhập Google.");
      }
    } catch (err) {
      setMailError(err.message || "Đăng nhập Google thất bại.");
      alert(`Lỗi đăng nhập Google: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const handleGoogleSignOut = async () => {
    setMailLoading(true);
    setMailError("");
    try {
      await logoutGoogleSignIn();
      await refreshMailState(currentSession);
      await refreshResources(currentSession);
    } catch (err) {
      setMailError(err.message || "Đăng xuất Google thất bại.");
      alert(`Lỗi đăng xuất Google: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const handleMailDisconnect = async () => {
    const sessionId = currentSession;
    if (!sessionId) return;
    if (!canManageMail) {
      const msg = "Vui lòng đăng nhập Google trước khi ngắt kết nối Gmail.";
      setMailError(msg);
      alert(msg);
      return;
    }
    setMailLoading(true);
    setMailError("");
    try {
      await disconnectMail(sessionId);
      await refreshMailState(sessionId);
    } catch (err) {
      setMailError(err.message || "Ngắt kết nối Gmail thất bại.");
      alert(`Lỗi ngắt kết nối Gmail: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const handleMailPollNow = async () => {
    const sessionId = currentSession;
    if (!sessionId) return;
    if (!canManageMail) {
      const msg = "Vui lòng đăng nhập Google trước khi poll mail.";
      setMailError(msg);
      alert(msg);
      return;
    }
    if (!mailConnected) {
      const msg = "Vui lòng bấm Connect Gmail trước khi poll mail.";
      setMailError(msg);
      alert(msg);
      return;
    }
    setMailLoading(true);
    setMailError("");
    try {
      await pollMailNow(sessionId);
      await refreshMailState(sessionId);
      await refreshResources(sessionId);
    } catch (err) {
      setMailError(err.message || "Poll mail thất bại.");
      alert(`Lỗi poll mail: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const handleSaveWhitelist = async () => {
    const sessionId = currentSession;
    if (!sessionId) return;
    if (!canManageMail) {
      const msg = "Vui lòng đăng nhập Google trước khi lưu whitelist.";
      setMailError(msg);
      alert(msg);
      return;
    }
    const senders = mailWhitelistText
      .split("\n")
      .map((v) => v.trim())
      .filter(Boolean);
    setMailLoading(true);
    setMailError("");
    try {
      await setMailWhitelist(sessionId, senders);
      await refreshMailState(sessionId);
    } catch (err) {
      setMailError(err.message || "Lưu whitelist thất bại.");
      alert(`Lỗi lưu whitelist: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const handleApplyCandidate = async (candidateId) => {
    const sessionId = currentSession;
    if (!sessionId) return;
    if (!canManageMail) {
      const msg = "Vui lòng đăng nhập Google trước khi apply candidate.";
      setMailError(msg);
      alert(msg);
      return;
    }
    setMailLoading(true);
    setMailError("");
    try {
      await applyMailCandidate(sessionId, candidateId);
      await refreshMailState(sessionId);
      await refreshResources(sessionId);
    } catch (err) {
      setMailError(err.message || "Apply candidate thất bại.");
      alert(`Lỗi apply candidate: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const handleRejectCandidate = async (candidateId) => {
    const sessionId = currentSession;
    if (!sessionId) return;
    if (!canManageMail) {
      const msg = "Vui lòng đăng nhập Google trước khi reject candidate.";
      setMailError(msg);
      alert(msg);
      return;
    }
    setMailLoading(true);
    setMailError("");
    try {
      await rejectMailCandidate(sessionId, candidateId, "manual_reject");
      await refreshMailState(sessionId);
    } catch (err) {
      setMailError(err.message || "Reject candidate thất bại.");
      alert(`Lỗi reject candidate: ${err.message}`);
    } finally {
      setMailLoading(false);
    }
  };

  const toggleCandidateExpanded = (candidateId) => {
    setExpandedMailCandidates((prev) => ({ ...prev, [candidateId]: !prev[candidateId] }));
  };

  const getGmailOpenUrl = (candidate, mailboxEmail) => {
    const mailbox = String(mailboxEmail || "").trim();
    // Gmail may rewrite authuser=email into /mail/u/<email>/... and return Temporary Error (404).
    // Keep authuser only when it is a numeric account index (0,1,2...), otherwise open in current profile account.
    const query = /^\d+$/.test(mailbox) ? `?authuser=${encodeURIComponent(mailbox)}` : "";
    const base = `https://mail.google.com/mail/${query}`;
    const threadId = String(candidate?.thread_id || "").trim();
    if (threadId) {
      return `${base}#inbox/${encodeURIComponent(threadId)}`;
    }
    const messageId = String(candidate?.message_id || "").trim();
    if (messageId) {
      return `${base}#all/${encodeURIComponent(messageId)}`;
    }
    return "";
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
    if (authState.authenticated) {
      updateChatSessionApi(currentSession, {
        selected_program_id: selected,
        selected_file_ids: selectedFileIds,
      }).catch((err) => console.error("Update server chat program failed", err));
    }
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

    const normalizedQuery = normalizeQueryForIntent(query);
    const transcriptNeedPattern = /(bang diem|tin chi|gpa|lap lich|lich hoc|mon con thieu|con thieu mon|thieu mon|hoc ky sau)/;
    if (!selectedFileIds.length && transcriptNeedPattern.test(normalizedQuery)) {
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
      const citations = Array.isArray(response?.citations) ? response.citations : [];
      updateMessages(sessionId, (prev) => [...prev, { type: "bot", text: answer, citations }]);
      const updatedHist = await fetchHistory(sessionId);
      setHistoryList(updatedHist);
      if (authState.authenticated) {
        await refreshChatSessions();
        const serverMessages = await fetchChatMessages(sessionId)
          .then((data) => (data?.messages || []).map(normalizeChatMessage).filter((msg) => msg.text))
          .catch(() => []);
        if (serverMessages.length) {
          setMessagesBySession((prev) => ({ ...prev, [sessionId]: serverMessages }));
        }
      }
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

        <div className="profile" style={{ alignItems: "stretch" }}>
          <div className="avatar">
            {((authState?.name || authState?.email || "U").trim().charAt(0) || "U").toUpperCase()}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: "14px", fontWeight: "600" }}>
              {authState?.authenticated ? (authState?.name || "Google User") : "Guest User"}
            </div>
            <div
              style={{
                fontSize: 12,
                color: "var(--text-secondary)",
                marginTop: 2,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
              title={authState?.authenticated ? (authState?.email || "") : "Chưa đăng nhập Google"}
            >
              {authState?.authenticated ? (authState?.email || "Đã đăng nhập") : "Chưa đăng nhập Google"}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
              Auth: {isGoogleSignedIn ? "Google" : "Guest"} | Mail: {mailConnected ? "Connected" : "Not connected"}
            </div>
            <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
              <button
                className="chip-btn"
                onClick={handleGoogleSignIn}
                disabled={mailLoading || authState?.authenticated}
                style={{ flex: 1, justifyContent: "center", fontSize: 12, padding: "6px 8px" }}
                title="Đăng nhập Google"
              >
                <i className="fab fa-google"></i> Sign in
              </button>
              <button
                className="chip-btn"
                onClick={handleGoogleSignOut}
                disabled={mailLoading || !authState?.authenticated}
                style={{ flex: 1, justifyContent: "center", fontSize: 12, padding: "6px 8px" }}
                title="Đăng xuất Google"
              >
                <i className="fas fa-sign-out-alt"></i> Sign out
              </button>
            </div>
          </div>
        </div>
      </aside>

      <main className="main">
        {showResourcePanel && (
          <div className="resource-panel" style={{
            position: "absolute",
            top: 20, right: 20, bottom: 20, width: "min(380px, calc(100% - 40px))",
            background: "var(--bg-secondary)",
            border: "1px solid var(--glass-border)",
            borderRadius: 12,
            backdropFilter: "blur(20px)",
            padding: 20,
            zIndex: 100,
            display: "flex", flexDirection: "column",
            boxShadow: "0 4px 30px rgba(0,0,0,0.5)",
            overflow: "hidden",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 15 }}>
              <h3 style={{ margin: 0 }}>Tài nguyên RAG</h3>
              <button className="icon-btn" onClick={() => setShowResourcePanel(false)}><i className="fas fa-times"></i></button>
            </div>

            <div className="resource-panel-scroll">
            <div>
              <div style={{ marginBottom: 10, fontSize: 13, fontWeight: "bold" }}>Thêm Resource Local (PDF/HTML)</div>
              <input
                type="file"
                ref={resourceUploadInputRef}
                accept="application/pdf,.pdf,text/html,.html,.htm"
                multiple
                style={{ display: "none" }}
                onChange={handleResourceFilesUpload}
              />
              <button className="chip-btn" onClick={() => resourceUploadInputRef.current?.click()} style={{ width: "100%", justifyContent: "center" }}>
                <i className="fas fa-upload"></i> Upload Resource
              </button>
              <div style={{ marginTop: 6, color: "#64748b", fontSize: 12 }}>
                Bạn có thể chọn nhiều file PDF/HTML cùng lúc. Hệ thống tự phân loại theo đuôi file.
              </div>

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

                <div style={{ marginTop: 18, paddingTop: 12, borderTop: "1px solid var(--glass-border)" }}>
                <div style={{ marginBottom: 10, fontSize: 13, fontWeight: "bold" }}>Mail Updates (Review-first)</div>
                <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                  <button className="chip-btn" onClick={handleMailPollNow} disabled={mailLoading || !canManageMail || !mailConnected} style={{ flex: 1 }}>
                    <i className={`fas ${mailLoading ? "fa-circle-notch fa-spin" : "fa-sync-alt"}`}></i> Poll now
                  </button>
                  <button className="chip-btn" onClick={handleSaveWhitelist} disabled={mailLoading || !canManageMail} style={{ flex: 1 }}>
                    <i className="fas fa-save"></i> Save whitelist
                  </button>
                </div>
                {!canManageMail && (
                  <div style={{ color: "#fbbf24", fontSize: 12, marginBottom: 8 }}>
                    Đăng nhập Google ở khung User để bật tính năng Mail Updates.
                  </div>
                )}
                <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 8 }}>
                  {mailStatusText}
                </div>
                <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                  <button
                    className="chip-btn"
                    onClick={handleMailConnect}
                    disabled={mailLoading || !canManageMail || mailConnected}
                    style={{ flex: 1 }}
                    title="Cấp quyền Gmail readonly để hệ thống có thể poll mail học vụ"
                  >
                    <i className="fas fa-link"></i> Connect Gmail
                  </button>
                  <button
                    className="chip-btn"
                    onClick={handleMailDisconnect}
                    disabled={mailLoading || !canManageMail || !mailConnected}
                    style={{ flex: 1 }}
                    title="Ngắt kết nối Gmail khỏi Mail Updates"
                  >
                    <i className="fas fa-unlink"></i> Disconnect
                  </button>
                </div>
                {canManageMail && !mailConnected && (
                  <div style={{ color: "#fbbf24", fontSize: 12, marginBottom: 8 }}>
                    Bạn đã đăng nhập Google nhưng chưa cấp quyền Gmail cho Mail Updates. Bấm Connect Gmail trước khi Poll now.
                  </div>
                )}
                {mailError && (
                  <div
                    style={{
                      color: "#fca5a5",
                      fontSize: 12,
                      marginBottom: 8,
                      padding: "6px 8px",
                      background: "rgba(127,29,29,0.22)",
                      border: "1px solid rgba(248,113,113,0.18)",
                      borderRadius: 6,
                    }}
                  >
                    {mailError}
                  </div>
                )}
                <textarea
                  value={mailWhitelistText}
                  onChange={(e) => setMailWhitelistText(e.target.value)}
                  rows={3}
                  placeholder={"Whitelist senders/domains (mỗi dòng 1 giá trị)\nvd: daotao@uet.edu.vn\nuet.edu.vn"}
                  style={{
                    width: "100%",
                    resize: "vertical",
                    background: "rgba(0,0,0,0.2)",
                    border: "1px solid var(--glass-border)",
                    borderRadius: 6,
                    padding: "8px 10px",
                    color: "white",
                    fontSize: 12,
                    marginBottom: 10,
                  }}
                />
                <div className="mail-candidate-list">
                  {(mailCandidates || []).map((item) => {
                    const openMailUrl = getGmailOpenUrl(item, mailStatus?.email || authState?.email || "");
                    return (
                    <div key={item.id} className="mail-candidate-card">
                      <div className="mail-candidate-subject">{item.subject || "(No subject)"}</div>
                      <div className="mail-candidate-meta">
                        {item.sender_email || "unknown sender"} | artifacts: {(item.artifacts || []).length}
                      </div>
                      <div className="mail-candidate-classification">
                        intent: {item?.classification?.intent || item?.intent || "other"} | confidence:{" "}
                        {Number(item?.classification?.confidence ?? item?.confidence ?? 0).toFixed(2)} | source:{" "}
                        {item?.classification?.source || "rule"}
                      </div>

                      {!!item.snippet && (
                        <div className="mail-candidate-preview">
                          <strong>Preview:</strong> {item.snippet}
                        </div>
                      )}

                      <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
                        <button
                          className="chip-btn"
                          onClick={() => toggleCandidateExpanded(item.id)}
                          style={{ flex: 1 }}
                          title="Xem đầy đủ nội dung để review trước khi Apply"
                        >
                          <i className={`fas ${expandedMailCandidates[item.id] ? "fa-chevron-up" : "fa-chevron-down"}`}></i>{" "}
                          {expandedMailCandidates[item.id] ? "Ẩn chi tiết" : "Xem chi tiết"}
                        </button>
                      </div>

                      {expandedMailCandidates[item.id] && (
                        <div className="mail-candidate-details">
                          {!!item.sender_display && (
                            <div className="mail-candidate-detail-row">
                              <strong>From:</strong> {item.sender_display}
                            </div>
                          )}
                          {!!item.body_preview && (
                            <div className="mail-candidate-detail-row">
                              <strong>Body:</strong>
                              <div className="mail-candidate-detail-text">{item.body_preview}</div>
                            </div>
                          )}
                          {Array.isArray(item?.classification?.reasons) && item.classification.reasons.length > 0 && (
                            <div className="mail-candidate-detail-row">
                              <strong>Match reasons:</strong>
                              <div className="mail-candidate-detail-text">
                                {item.classification.reasons.join(" | ")}
                              </div>
                            </div>
                          )}
                          {Array.isArray(item?.artifacts) && item.artifacts.length > 0 && (
                            <div className="mail-candidate-detail-row">
                              <strong>Artifacts:</strong>
                              <div className="mail-candidate-detail-text">
                                {item.artifacts
                                  .map((artifact) => artifact?.name || artifact?.url || artifact?.type || "artifact")
                                  .join(" | ")}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      <div style={{ display: "flex", gap: 6 }}>
                        {!!openMailUrl && (
                          <a
                            className="chip-btn"
                            href={openMailUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ flex: 1, textAlign: "center", textDecoration: "none" }}
                            title={`Mở email gốc trong Gmail (${mailStatus?.email || authState?.email || "current account"})`}
                          >
                            <i className="fas fa-external-link-alt"></i> Open Gmail
                          </a>
                        )}
                        <button className="chip-btn" onClick={() => handleApplyCandidate(item.id)} disabled={mailLoading || !canManageMail} style={{ flex: 1 }}>
                          <i className="fas fa-check"></i> Apply
                        </button>
                        <button className="chip-btn" onClick={() => handleRejectCandidate(item.id)} disabled={mailLoading || !canManageMail} style={{ flex: 1 }}>
                          <i className="fas fa-times"></i> Reject
                        </button>
                      </div>
                    </div>
                  );})}
                  {(!mailCandidates || mailCandidates.length === 0) && (
                    <div style={{ color: "#64748b", fontSize: 12 }}>
                      {mailLoading
                        ? "Đang tải trạng thái mail..."
                        : authState.loading
                          ? "Đang kiểm tra đăng nhập Google..."
                          : "Không có candidate pending."}
                    </div>
                  )}
                </div>
              </div>

              <div style={{ marginTop: 18, paddingTop: 12, borderTop: "1px solid var(--glass-border)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, gap: 8 }}>
                  <button
                    className="chip-btn"
                    onClick={() => setResourceListVisible((v) => !v)}
                    style={{ flex: 1, justifyContent: "center" }}
                  >
                    <i className={`fas ${resourceListVisible ? "fa-folder-open" : "fa-folder"}`}></i>{" "}
                    {resourceListVisible ? "Ẩn danh sách tài nguyên" : "Hiện danh sách tài nguyên"}
                  </button>
                  <div style={{ fontSize: 11, color: "#94a3b8", whiteSpace: "nowrap" }}>
                    {filteredResources.length}/{resources.length}
                  </div>
                </div>
                {resourceListVisible && (
                  <>
                    <div className="resource-search-wrap">
                      <i className="fas fa-search resource-search-icon"></i>
                      <input
                        value={resourceSearch}
                        onChange={(e) => setResourceSearch(e.target.value)}
                        placeholder="Tìm theo tên file..."
                        className="resource-search-input"
                      />
                    </div>
                    <div className="resource-list-compact">
                      {resourceLoading && (
                        <div style={{ textAlign: "center", color: "#94a3b8", fontSize: 12 }}>
                          <i className="fas fa-circle-notch fa-spin"></i> Loading...
                        </div>
                      )}
                      {!resourceLoading && visibleResources.map((res, i) => (
                        <div key={res.id || i} className="resource-item-card">
                          <i
                            className={`fas ${res.type === "url" ? "fa-globe" : "fa-file-pdf"} resource-item-icon`}
                            style={{ color: res.type === "url" ? "#60a5fa" : "#f87171" }}
                          ></i>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div className="resource-item-name" title={res.name}>{res.name}</div>
                            <div style={{ marginTop: 4, display: "flex", gap: 6 }}>
                              <span
                                style={{
                                  fontSize: 11,
                                  padding: "2px 6px",
                                  borderRadius: 999,
                                  background:
                                    res.scope === "global"
                                      ? "rgba(96,165,250,0.18)"
                                      : res.scope === "user"
                                        ? "rgba(250,204,21,0.18)"
                                        : "rgba(74,222,128,0.18)",
                                  color:
                                    res.scope === "global"
                                      ? "#93c5fd"
                                      : res.scope === "user"
                                        ? "#fde68a"
                                        : "#86efac",
                                  border: "1px solid rgba(255,255,255,0.08)",
                                }}
                              >
                                {res.scope === "global" ? "Global" : res.scope === "user" ? "Private" : "Session"}
                              </span>
                            </div>
                          </div>
                          <button
                            onClick={() => handleDeleteResource(res.id)}
                            style={{ background: "transparent", border: "none", color: "#94a3b8", cursor: "pointer", padding: "0 5px" }}
                            title="Xóa tài nguyên"
                          >
                            <i className="fas fa-trash-alt"></i>
                          </button>
                        </div>
                      ))}
                      {!resourceLoading && filteredResources.length === 0 && (
                        <div style={{ color: "#64748b", fontSize: 12 }}>
                          {resources.length === 0 ? "Chưa có tài nguyên nào." : "Không tìm thấy tài nguyên phù hợp."}
                        </div>
                      )}
                    </div>
                    {!resourceLoading && filteredResources.length > resourcePreviewLimit && (
                      <button
                        className="chip-btn"
                        onClick={() => setResourceListExpanded((v) => !v)}
                        style={{ width: "100%", justifyContent: "center", marginTop: 8 }}
                      >
                        <i className={`fas ${resourceListExpanded ? "fa-chevron-up" : "fa-chevron-down"}`}></i>{" "}
                        {resourceListExpanded
                          ? "Thu gọn danh sách"
                          : `Xem thêm ${filteredResources.length - resourcePreviewLimit} file`}
                      </button>
                    )}
                  </>
                )}
              </div>
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
                      {(() => {
                        const displayText = stripSourceFooterFromMessage(msg.text);
                        const citations = Array.isArray(msg.citations) ? msg.citations : [];
                        const resolveLineCitations = (lineText, maxMatches = 2) => {
                          const normalizedLine = String(lineText || "").trim();
                          if (!normalizedLine || !citations.length) return [];
                          if (shouldSkipCitationLine(normalizedLine)) return [];
                          return matchLineCitations(normalizedLine, citations, maxMatches);
                        };
                        const markdownComponents = citations.length
                          ? {
                              li: ({ children, ...props }) => {
                                const lineText = extractNodePlainText(children);
                                const matchedCitations = resolveLineCitations(lineText, 2);
                                return (
                                  <li {...props}>
                                    <span className="md-line-source-wrap">
                                      <span className="md-line-source-text">{children}</span>
                                      {matchedCitations.length > 0 && (
                                        <span className="md-line-source-badges">
                                          {matchedCitations.map((citation, matchIndex) => (
                                            <button
                                              key={`${citation?.id || matchIndex}-${citation?.source_file || "source"}-${matchIndex}`}
                                              className="md-line-source-badge"
                                              onClick={() => setCitationViewer(citation)}
                                              title={citationDisplayLabel(citation, matchIndex)}
                                            >
                                              {citationAnchorLabel(citation, matchIndex)}
                                            </button>
                                          ))}
                                        </span>
                                      )}
                                    </span>
                                  </li>
                                );
                              },
                              p: ({ children, ...props }) => {
                                const lineText = extractNodePlainText(children);
                                const matchedCitations = resolveLineCitations(lineText, 1);
                                return (
                                  <p {...props}>
                                    <span className="md-line-source-wrap">
                                      <span className="md-line-source-text">{children}</span>
                                      {matchedCitations.length > 0 && (
                                        <span className="md-line-source-badges">
                                          {matchedCitations.map((citation, matchIndex) => (
                                            <button
                                              key={`${citation?.id || matchIndex}-${citation?.source_file || "source"}-p-${matchIndex}`}
                                              className="md-line-source-badge"
                                              onClick={() => setCitationViewer(citation)}
                                              title={citationDisplayLabel(citation, matchIndex)}
                                            >
                                              {citationAnchorLabel(citation, matchIndex)}
                                            </button>
                                          ))}
                                        </span>
                                      )}
                                    </span>
                                  </p>
                                );
                              },
                            }
                          : undefined;

                        return (
                          <>
                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                              {displayText}
                            </ReactMarkdown>
                          </>
                        );
                      })()}
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
            <div className="message-wrapper loading-response-wrapper">
              <div className="msg-avatar bot">
                <i className="fas fa-bolt"></i>
              </div>
              <div className="msg-content bot-text loading-response-bubble">
                <div className="loading-response-text">
                  Đang suy nghĩ{".".repeat(loadingFrame + 1)}
                </div>
                <div className="loading-dot-row" aria-live="polite" aria-label="Assistant is typing">
                  <span className={`loading-dot ${loadingFrame === 0 ? "active" : ""}`}>•</span>
                  <span className={`loading-dot ${loadingFrame === 1 ? "active" : ""}`}>•</span>
                  <span className={`loading-dot ${loadingFrame === 2 ? "active" : ""}`}>•</span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef}></div>
        </div>

        {citationViewer && (
          <div className="citation-viewer-backdrop" onClick={() => setCitationViewer(null)}>
            <div className="citation-viewer-card" onClick={(e) => e.stopPropagation()}>
              <div className="citation-viewer-head">
                <strong>{citationDisplayLabel(citationViewer, 0)}</strong>
                <button className="icon-btn" onClick={() => setCitationViewer(null)} title="Đóng">
                  <i className="fas fa-times"></i>
                </button>
              </div>
              <div className="citation-viewer-body">
                <pre>{String(citationViewer?.excerpt || "(Không có nội dung nguồn)")}</pre>
              </div>
            </div>
          </div>
        )}

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
