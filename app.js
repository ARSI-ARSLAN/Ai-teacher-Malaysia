/**
 * app.js - AI Guru Matematik Chat Interface
 * Handles: SSE streaming, language toggle, form selector, MathJax rendering
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
    language: 'bm',
    formFilter: '',
    isStreaming: false,
    messageCount: 0,
};

// ── UI Translations ────────────────────────────────────────────────────────
const UI_TEXT = {
    bm: {
        appTitle: 'AI Guru Matematik',
        appSubtitle: 'Sistem Pengajaran Berteraskan KSSM',
        welcomeTitle: 'Selamat Datang!',
        welcomeDesc: 'Saya ialah AI Guru Matematik anda. Tanya saya apa sahaja tentang Matematik KSSM Tingkatan 1–5.',
        placeholder: 'Taip soalan matematik anda di sini...',
        footerNote: 'AI ini berdasarkan buku teks KSSM • Jawapan mungkin tidak sempurna',
        thinking: 'Berfikir...',
        samples: [
            'Apakah Teorem Pythagoras?',
            'Selesaikan: 2x + 5 = 13',
            'Kirakan luas segi tiga dengan tapak 6cm dan tinggi 8cm',
            'Apakah itu persamaan kuadratik?',
        ],
        statusOnline: 'Sistem dalam talian • Siap membantu',
        statusError: 'Ralat sambungan',
    },
    en: {
        appTitle: 'AI Math Teacher',
        appSubtitle: 'KSSM Curriculum-Based Learning System',
        welcomeTitle: 'Welcome!',
        welcomeDesc: 'I am your AI Math Teacher, trained on KSSM textbooks for Forms 1–5. Ask me anything!',
        placeholder: 'Type your math question here...',
        footerNote: 'AI is based on KSSM textbooks • Answers may not be perfect',
        thinking: 'Thinking...',
        samples: [
            'What is Pythagoras Theorem?',
            'Solve: 2x + 5 = 13, show steps',
            'What is the quadratic formula?',
            'Explain simultaneous equations',
        ],
        statusOnline: 'System online • Ready to help',
        statusError: 'Connection error',
    },
};

// ── DOM Refs ────────────────────────────────────────────────────────────────
const chatWindow = document.getElementById('chat-window');
const messagesEl = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcome-screen');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const charCount = document.getElementById('char-count');
const formSelect = document.getElementById('form-select');
const langBm = document.getElementById('lang-bm');
const langEn = document.getElementById('lang-en');

// ── Language Toggle ─────────────────────────────────────────────────────────
function setLanguage(lang) {
    state.language = lang;
    const t = UI_TEXT[lang];

    // Toggle active classes
    langBm.classList.toggle('active', lang === 'bm');
    langEn.classList.toggle('active', lang === 'en');

    // Update UI text
    document.getElementById('app-title').textContent = t.appTitle;
    document.getElementById('app-subtitle').textContent = t.appSubtitle;
    document.getElementById('welcome-title').textContent = t.welcomeTitle;
    document.getElementById('welcome-desc').textContent = t.welcomeDesc;
    questionInput.placeholder = t.placeholder;
    document.getElementById('footer-note').textContent = t.footerNote;

    // Update sample questions
    const samplesEl = document.getElementById('sample-questions');
    if (samplesEl) {
        samplesEl.innerHTML = t.samples
            .map(
                (q) =>
                    `<button class="sample-btn" onclick="useQuestion(this)">${q}</button>`
            )
            .join('');
    }
}

document.getElementById('lang-toggle').addEventListener('click', (e) => {
    const newLang = state.language === 'bm' ? 'en' : 'bm';
    setLanguage(newLang);
});

// ── Form Selector ───────────────────────────────────────────────────────────
formSelect.addEventListener('change', () => {
    state.formFilter = formSelect.value;
});

// ── Sample Question ──────────────────────────────────────────────────────────
function useQuestion(btn) {
    questionInput.value = btn.textContent;
    updateCharCount();
    questionInput.focus();
}
window.useQuestion = useQuestion;

// ── Textarea Auto-resize + Char Count ───────────────────────────────────────
function updateCharCount() {
    const len = questionInput.value.length;
    charCount.textContent = `${len}/2000`;
    charCount.style.color = len > 1800 ? '#fc8181' : 'var(--text-muted)';
}

questionInput.addEventListener('input', () => {
    updateCharCount();
    autoResize();
});

function autoResize() {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
}

// Send on Enter (Shift+Enter for newline)
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

// ── Message Rendering ────────────────────────────────────────────────────────
function createUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
    <div class="msg-avatar">👤</div>
    <div class="msg-content">
      <div class="msg-bubble">${escapeHtml(text)}</div>
      <div class="msg-meta">${getTime()}</div>
    </div>
  `;
    return div;
}

function createAiMessage() {
    const id = `msg-ai-${Date.now()}`;
    const div = document.createElement('div');
    div.className = 'message ai';
    div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble streaming" id="${id}"></div>
      <div class="msg-meta" id="${id}-meta">${getTime()}</div>
    </div>
  `;
    return { element: div, bubbleId: id };
}

function appendMessage(el) {
    if (welcomeScreen && welcomeScreen.style.display !== 'none') {
        welcomeScreen.style.display = 'none';
    }
    messagesEl.appendChild(el);
    scrollToBottom();
}

function scrollToBottom() {
    chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: 'smooth' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

function getTime() {
    return new Date().toLocaleTimeString('en-MY', {
        hour: '2-digit', minute: '2-digit',
    });
}

// ── Send Message & SSE Stream ────────────────────────────────────────────────
async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || state.isStreaming) return;

    state.isStreaming = true;
    state.messageCount++;

    // Clear input
    questionInput.value = '';
    autoResize();
    updateCharCount();
    sendBtn.disabled = true;

    // Show user message
    appendMessage(createUserMessage(question));

    // Show typing indicator
    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    // Create AI message bubble
    const { element: aiMsgEl, bubbleId } = createAiMessage();
    appendMessage(aiMsgEl);
    const bubble = document.getElementById(bubbleId);
    bubble.textContent = UI_TEXT[state.language].thinking;

    let fullText = '';
    let firstToken = true;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                language: 'auto',   // always auto-detect from question text
                form_filter: state.formFilter || null,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        typingIndicator.classList.add('hidden');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // keep incomplete chunk

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.done) {
                        bubble.classList.remove('streaming');
                        renderMath(bubble);
                        break;
                    }

                    if (data.error) {
                        bubble.textContent = `❌ ${data.error}`;
                        bubble.classList.remove('streaming');
                        break;
                    }

                    if (data.token) {
                        if (firstToken) {
                            bubble.textContent = '';
                            firstToken = false;
                        }
                        fullText += data.token;
                        bubble.textContent = fullText;
                        scrollToBottom();
                    }
                } catch (_) { /* skip malformed */ }
            }
        }

    } catch (err) {
        typingIndicator.classList.add('hidden');
        bubble.textContent = `❌ Connection error: ${err.message}`;
        bubble.classList.remove('streaming');
        console.error('Chat error:', err);
    } finally {
        state.isStreaming = false;
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

// ── MathJax Rendering ────────────────────────────────────────────────────────
function renderMath(element) {
    if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise([element]).catch(console.warn);
    }
}

// ── Health Check & Status ────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.status === 'ok') {
            statusDot.className = 'status-dot online';
            statusText.textContent = UI_TEXT[state.language].statusOnline;
            statusText.textContent += ` | ${data.total_chunks?.toLocaleString() ?? 0} chunks`;
        } else {
            throw new Error(data.detail);
        }
    } catch (err) {
        statusDot.className = 'status-dot error';
        statusText.textContent = UI_TEXT[state.language].statusError;
        console.warn('Health check failed:', err);
    }
}

// ── MathJax Config ───────────────────────────────────────────────────────────
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    },
};

// ── Init ─────────────────────────────────────────────────────────────────────
setLanguage('bm');
checkHealth();
setInterval(checkHealth, 30000); // ping every 30s
