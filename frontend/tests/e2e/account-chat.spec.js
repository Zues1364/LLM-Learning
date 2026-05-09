import { expect, test } from "@playwright/test";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "http://127.0.0.1:5173",
  "Access-Control-Allow-Credentials": "true",
  "Access-Control-Allow-Headers": "content-type",
  "Access-Control-Allow-Methods": "GET,POST,PATCH,DELETE,OPTIONS",
};

const users = {
  alice: {
    id: "user-a",
    email: "alice@vnu.edu.vn",
    name: "Alice Nguyen",
  },
  bob: {
    id: "user-b",
    email: "bob@vnu.edu.vn",
    name: "Bob Tran",
  },
};

const programPayload = {
  programs: [
    {
      id: "ckt_2025",
      name: "Co ky thuat",
      display_name: "Co ky thuat (QH-2025)",
      qh_label: "QH-2025",
      group_name: "Co ky thuat",
      year_end: 2025,
    },
    {
      id: "it_2025",
      name: "Cong nghe thong tin",
      display_name: "Cong nghe thong tin (QH-2025)",
      qh_label: "QH-2025",
      group_name: "Cong nghe thong tin",
      year_end: 2025,
    },
    {
      id: "cs_2022",
      name: "Khoa hoc may tinh",
      display_name: "Khoa hoc may tinh (QH-2022-2024)",
      qh_label: "QH-2022-2024",
      group_name: "Khoa hoc may tinh",
      year_end: 2024,
    },
  ],
};

function chatSession(id, title) {
  return {
    id,
    title,
    selected_program_id: "cs_2022",
    selected_file_ids: [],
  };
}

function textMessage(role, content) {
  return { role, content, citations: [] };
}

async function fulfillJson(route, body, status = 200) {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: CORS_HEADERS,
    body: JSON.stringify(body),
  });
}

async function setupApiMock(page, state) {
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const isApi =
      (url.hostname === "127.0.0.1" || url.hostname === "localhost") &&
      url.port === "9000";

    if (!isApi) {
      await route.continue();
      return;
    }

    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: CORS_HEADERS });
      return;
    }

    const path = url.pathname;
    const userId = state.user?.id || "";

    if (path === "/api/auth/me") {
      await fulfillJson(route, {
        authenticated: Boolean(state.authenticated),
        user: state.authenticated ? state.user : null,
      });
      return;
    }

    if (path === "/api/auth/logout") {
      state.authenticated = false;
      state.user = null;
      await fulfillJson(route, { ok: true });
      return;
    }

    if (path === "/api/programs") {
      await fulfillJson(route, programPayload);
      return;
    }

    if (path === "/files") {
      await fulfillJson(route, [
        {
          file_id: "transcript.pdf",
          file_name: "transcript.pdf",
        },
      ]);
      return;
    }

    if (path === "/history") {
      await fulfillJson(route, []);
      return;
    }

    if (path === "/api/resources") {
      await fulfillJson(route, []);
      return;
    }

    if (path === "/api/mail/status") {
      await fulfillJson(route, {
        mode: state.authenticated ? "user" : "session",
        connected: false,
        connected_email: "",
        pending_count: 0,
      });
      return;
    }

    if (path === "/api/mail/whitelist") {
      await fulfillJson(route, { senders: [] });
      return;
    }

    if (path === "/api/mail/candidates") {
      await fulfillJson(route, { candidates: [] });
      return;
    }

    if (path === "/api/chat/sessions" && request.method() === "GET") {
      state.chatSessionsGetCount = (state.chatSessionsGetCount || 0) + 1;
      const sessionsForResponse = JSON.parse(JSON.stringify(state.serverSessionsByUser[userId] || []));
      if (
        state.delayChatSessionsFromRequest &&
        state.chatSessionsGetCount >= state.delayChatSessionsFromRequest
      ) {
        await new Promise((resolve) => setTimeout(resolve, state.delayChatSessionsMs || 0));
      }
      await fulfillJson(route, {
        sessions: sessionsForResponse,
      });
      return;
    }

    if (path === "/ask" && request.method() === "POST") {
      const payload = request.postDataJSON();
      state.askRequests ||= [];
      state.askRequests.push(payload);
      if (state.authenticated) {
        state.serverMessagesByUser[userId] ||= {};
        state.serverMessagesByUser[userId][payload.session_id] ||= [];
        state.serverMessagesByUser[userId][payload.session_id].push(textMessage("user", payload.query || ""));
        state.serverMessagesByUser[userId][payload.session_id].push(textMessage("assistant", "ok"));
      }
      await fulfillJson(route, {
        answer: "ok",
        selected_program_id: payload.program_id || "",
        citations: [],
      });
      return;
    }

    if (path === "/api/chat/migrate" && request.method() === "POST") {
      if (!state.authenticated) {
        await fulfillJson(route, { detail: "Not authenticated" }, 401);
        return;
      }
      const payload = request.postDataJSON();
      const sessions = Array.isArray(payload?.sessions) ? payload.sessions : [];
      state.migrationRequests.push(sessions);

      state.serverSessionsByUser[userId] ||= [];
      state.serverMessagesByUser[userId] ||= {};

      let importedMessages = 0;
      for (const item of sessions) {
        const id = String(item?.session_id || "").trim();
        if (!id) continue;
        const existing = state.serverSessionsByUser[userId].some((session) => session.id === id);
        if (!existing) {
          state.serverSessionsByUser[userId].push({
            id,
            title: String(item?.title || "Imported chat"),
            selected_program_id: item?.selected_program_id || "",
            selected_file_ids: Array.isArray(item?.selected_file_ids) ? item.selected_file_ids : [],
          });
        }
        const messages = Array.isArray(item?.messages) ? item.messages : [];
        state.serverMessagesByUser[userId][id] = messages.map((message) =>
          textMessage(message.role || "assistant", message.content || "")
        );
        importedMessages += messages.length;
      }

      await fulfillJson(route, {
        imported_sessions: sessions.length,
        imported_messages: importedMessages,
      });
      return;
    }

    const sessionMatch = path.match(/^\/api\/chat\/sessions\/([^/]+)$/);
    if (sessionMatch && request.method() === "PATCH") {
      const sessionId = decodeURIComponent(sessionMatch[1]);
      const payload = request.postDataJSON();
      const sessions = state.serverSessionsByUser[userId] || [];
      const session = sessions.find((item) => item.id === sessionId);
      if (!session) {
        await fulfillJson(route, { detail: "Not found" }, 404);
        return;
      }
      if (Object.prototype.hasOwnProperty.call(payload || {}, "title")) {
        session.title = payload.title || "Phien moi";
      }
      if (Object.prototype.hasOwnProperty.call(payload || {}, "selected_program_id")) {
        session.selected_program_id = payload.selected_program_id || "";
      }
      if (Object.prototype.hasOwnProperty.call(payload || {}, "selected_file_ids")) {
        session.selected_file_ids = Array.isArray(payload.selected_file_ids)
          ? payload.selected_file_ids
          : [];
      }
      await fulfillJson(route, { session });
      return;
    }

    if (sessionMatch && request.method() === "DELETE") {
      const sessionId = decodeURIComponent(sessionMatch[1]);
      state.serverSessionsByUser[userId] = (state.serverSessionsByUser[userId] || []).filter(
        (item) => item.id !== sessionId
      );
      await fulfillJson(route, { ok: true });
      return;
    }

    const match = path.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/);
    if (match && request.method() === "GET") {
      const sessionId = decodeURIComponent(match[1]);
      await fulfillJson(route, {
        messages: state.serverMessagesByUser[userId]?.[sessionId] || [],
      });
      return;
    }

    await fulfillJson(route, {});
  });
}

async function loginAs(page, state, user) {
  state.authenticated = true;
  state.user = user;
  await page.reload();
  await expect(page.getByText(user.name)).toBeVisible();
}

test("authenticated sessions are isolated across users and cleared on logout", async ({ page }) => {
  const state = {
    authenticated: false,
    user: null,
    migrationRequests: [],
    serverSessionsByUser: {
      [users.alice.id]: [chatSession("alice-session", "Alice graduation plan")],
      [users.bob.id]: [chatSession("bob-session", "Bob timetable review")],
    },
    serverMessagesByUser: {
      [users.alice.id]: {
        "alice-session": [textMessage("user", "alice private question")],
      },
      [users.bob.id]: {
        "bob-session": [textMessage("user", "bob private question")],
      },
    },
  };
  await setupApiMock(page, state);

  await page.goto("/");
  await expect(page.getByText("Guest User")).toBeVisible();
  await expect(page.getByText("Phiên mới")).toBeVisible();
  await expect(page.getByText("Alice graduation plan")).toHaveCount(0);
  await expect(page.getByText("Bob timetable review")).toHaveCount(0);

  await loginAs(page, state, users.alice);
  await expect(page.getByText("Alice graduation plan")).toBeVisible();
  await expect(page.getByText("Bob timetable review")).toHaveCount(0);

  await page.getByRole("button", { name: /Sign out/i }).click();
  await expect(page.getByText("Guest User")).toBeVisible();
  await expect(page.getByText("Alice graduation plan")).toHaveCount(0);
  await expect(page.getByText("Phiên mới")).toBeVisible();

  await loginAs(page, state, users.bob);
  await expect(page.getByText("Bob timetable review")).toBeVisible();
  await expect(page.getByText("Alice graduation plan")).toHaveCount(0);
});

test("server session refresh does not overwrite selected transcript files", async ({ page }) => {
  const state = {
    authenticated: true,
    user: users.alice,
    migrationRequests: [],
    askRequests: [],
    chatSessionsGetCount: 0,
    delayChatSessionsFromRequest: 2,
    delayChatSessionsMs: 800,
    serverSessionsByUser: {
      [users.alice.id]: [
        {
          id: "alice-session",
          title: "Alice transcript",
          selected_program_id: "cs_2022",
          selected_file_ids: [],
        },
      ],
    },
    serverMessagesByUser: {
      [users.alice.id]: {
        "alice-session": [],
      },
    },
  };
  await setupApiMock(page, state);

  await page.goto("/");
  await expect(page.getByText("Alice transcript")).toBeVisible();
  await expect(page.getByText("transcript.pdf")).toBeVisible();
  await expect.poll(() => state.chatSessionsGetCount, { timeout: 5_000 }).toBeGreaterThanOrEqual(2);

  await page.locator(".file-chip", { hasText: "transcript.pdf" }).click();
  await expect(page.locator(".file-chip.selected", { hasText: "transcript.pdf" })).toBeVisible();

  await page.waitForTimeout(state.delayChatSessionsMs + 150);
  await expect(page.locator(".file-chip.selected", { hasText: "transcript.pdf" })).toBeVisible();

  const query = "lieu voi so tin chi con lai toi co the len bang gioi khong";
  await page.locator("textarea").fill(query);
  await page.locator(".input-row .icon-btn").last().click();

  await expect.poll(() => state.askRequests.length, { timeout: 5_000 }).toBe(1);
  expect(state.askRequests[0].file_ids).toEqual(["transcript.pdf"]);
  expect(state.serverSessionsByUser[users.alice.id][0].selected_file_ids).toEqual(["transcript.pdf"]);
});

test("first question names placeholder session and keeps session actions visible", async ({ page }) => {
  const query = "toi con bao nhieu tin chi la co the tot nghiep voi chuong trinh khoa hoc may tinh";
  const expectedTitle = `${query.slice(0, 77).trimEnd()}...`;
  const state = {
    authenticated: true,
    user: users.alice,
    migrationRequests: [],
    askRequests: [],
    serverSessionsByUser: {
      [users.alice.id]: [
        {
          id: "new-session",
          title: "Phien 6",
          selected_program_id: "cs_2022",
          selected_file_ids: ["transcript.pdf"],
        },
      ],
    },
    serverMessagesByUser: {
      [users.alice.id]: {
        "new-session": [],
      },
    },
  };
  await setupApiMock(page, state);

  await page.goto("/");
  await expect(page.getByText("Phien 6")).toBeVisible();

  await page.locator("textarea").fill(query);
  await page.locator(".input-row .icon-btn").last().click();

  await expect.poll(() => state.askRequests.length, { timeout: 5_000 }).toBe(1);
  await expect(page.locator(".chat-session-item").filter({ hasText: expectedTitle })).toBeVisible();
  expect(state.serverSessionsByUser[users.alice.id][0].title).toBe(expectedTitle);

  const item = page.locator(".chat-session-item").filter({ hasText: expectedTitle }).first();
  const actions = item.locator(".session-action-btn");
  await expect(actions).toHaveCount(2);
  const itemBox = await item.boundingBox();
  const deleteBox = await actions.nth(1).boundingBox();
  expect(itemBox).not.toBeNull();
  expect(deleteBox).not.toBeNull();
  expect(deleteBox.x + deleteBox.width).toBeLessThanOrEqual(itemBox.x + itemBox.width + 1);

  await actions.nth(1).click();
  await expect(page.locator(".chat-session-item").filter({ hasText: expectedTitle })).toHaveCount(0);
});

test("legacy browser chat is migrated into the authenticated account once", async ({ page }) => {
  const legacySessionId = "legacy-local-session";
  const state = {
    authenticated: false,
    user: null,
    migrationRequests: [],
    serverSessionsByUser: {
      [users.alice.id]: [],
    },
    serverMessagesByUser: {
      [users.alice.id]: {},
    },
  };
  await setupApiMock(page, state);

  await page.addInitScript((sessionId) => {
    localStorage.setItem(
      "guestSessions",
      JSON.stringify([{ id: sessionId, title: "Local transcript planning" }])
    );
    localStorage.setItem("guestCurrentSession", JSON.stringify(sessionId));
    localStorage.setItem(
      "guestMessagesBySession",
      JSON.stringify({
        [sessionId]: [
          { type: "user", text: "toi con thieu mon nao" },
          { type: "bot", text: "ban con thieu khoa luan tot nghiep" },
        ],
      })
    );
    localStorage.setItem(
      "guestSelectedProgramBySession",
      JSON.stringify({ [sessionId]: "cs_2022" })
    );
    localStorage.setItem(
      "guestSelectedFilesBySession",
      JSON.stringify({ [sessionId]: ["transcript.pdf"] })
    );
  }, legacySessionId);

  await page.goto("/");
  await expect(page.getByText("Local transcript planning")).toBeVisible();

  await loginAs(page, state, users.alice);

  await expect
    .poll(() => state.migrationRequests.length, { timeout: 5_000 })
    .toBe(1);
  expect(state.migrationRequests[0][0]).toMatchObject({
    session_id: legacySessionId,
    title: "Local transcript planning",
    selected_program_id: "cs_2022",
    selected_file_ids: ["transcript.pdf"],
  });

  await expect(page.getByText("Local transcript planning")).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("guestSessions")), { timeout: 5_000 })
    .toBeNull();
});

test("server session refresh does not overwrite an in-progress program selection", async ({ page }) => {
  const state = {
    authenticated: true,
    user: users.alice,
    migrationRequests: [],
    chatSessionsGetCount: 0,
    delayChatSessionsFromRequest: 2,
    delayChatSessionsMs: 800,
    serverSessionsByUser: {
      [users.alice.id]: [
        {
          id: "alice-session",
          title: "Alice curriculum",
          selected_program_id: "ckt_2025",
          selected_file_ids: [],
        },
      ],
    },
    serverMessagesByUser: {
      [users.alice.id]: {
        "alice-session": [],
      },
    },
  };
  await setupApiMock(page, state);

  await page.goto("/");
  await expect(page.getByText("Alice curriculum")).toBeVisible();
  await expect(page.locator("select")).toHaveValue("ckt_2025");

  await expect.poll(() => state.chatSessionsGetCount, { timeout: 5_000 }).toBeGreaterThanOrEqual(2);
  await page.locator("select").selectOption("it_2025");
  await expect(page.locator("select")).toHaveValue("it_2025");

  await page.waitForTimeout(state.delayChatSessionsMs + 150);
  await expect(page.locator("select")).toHaveValue("it_2025");

  await page.locator("button.program-confirm-btn").click();
  await expect(page.getByText("Đã chọn chương trình đào tạo: Cong nghe thong tin (QH-2025)")).toBeVisible();
  expect(state.serverSessionsByUser[users.alice.id][0].selected_program_id).toBe("it_2025");
});
