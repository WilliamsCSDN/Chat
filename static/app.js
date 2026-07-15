/**
 * 百炼大模型对话 - 前端交互逻辑 (AG-UI 协议)
 */

// ==================== DOM 元素 ====================
const chatMessages = document.getElementById('chatMessages');
const welcomeScreen = document.getElementById('welcomeScreen');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const newChatBtn = document.getElementById('newChatBtn');
const modelSelect = document.getElementById('modelSelect');
const promptBtns = document.querySelectorAll('.prompt-btn');
const chatTabs = document.querySelectorAll('.chat-tab');
const normalTabPanel = document.getElementById('normalTabPanel');
const pdfTabPanel = document.getElementById('pdfTabPanel');
const pdfChatMessages = document.getElementById('pdfChatMessages');
const pdfMessageInput = document.getElementById('pdfMessageInput');
const pdfSendBtn = document.getElementById('pdfSendBtn');
const openaiMessagesEl = document.getElementById('openaiMessages');
const openaiModelCallsEl = document.getElementById('openaiModelCalls');

// ==================== 状态管理 ====================
let messages = [];
let pdfMessages = [];
let isGenerating = false;
let abortController = null;
let isPdfGenerating = false;
let activeTab = 'normal';
let currentThreadId = null;
let openaiCallIndex = 0;
const openaiCallElements = {};
let openaiMessagesEmpty = true;
let openaiCallsEmpty = true;

// ==================== 初始化 Marked ====================
marked.setOptions({
    highlight: function (code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true,
});

// ==================== 事件绑定 ====================

sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', stopGeneration);

chatTabs.forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

newChatBtn.addEventListener('click', () => {
    currentThreadId = null;
    if (activeTab === 'normal') {
        messages = [];
        chatMessages.innerHTML = '';
        chatMessages.appendChild(createWelcomeScreen());
        clearOpenAIInspector();
        messageInput.focus();
        return;
    }
    pdfMessages = [];
    pdfChatMessages.innerHTML = '';
    pdfChatMessages.appendChild(createPdfWelcomeScreen());
    pdfMessageInput.focus();
});

promptBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        messageInput.value = btn.dataset.prompt;
        sendMessage();
    });
});

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating) {
            sendMessage();
        }
    }
});

messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
});

if (pdfSendBtn) {
    pdfSendBtn.addEventListener('click', sendPdfMessage);
}

if (pdfMessageInput) {
    pdfMessageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isPdfGenerating) {
                sendPdfMessage();
            }
        }
    });
    pdfMessageInput.addEventListener('input', () => {
        pdfMessageInput.style.height = 'auto';
        pdfMessageInput.style.height = Math.min(pdfMessageInput.scrollHeight, 200) + 'px';
    });
}

// ==================== 核心函数 ====================

async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || isGenerating) return;

    const welcome = document.getElementById('welcomeScreen');
    if (welcome) {
        welcome.remove();
    }

    messages.push({ role: 'user', content });
    appendMessage('user', content);

    messageInput.value = '';
    messageInput.style.height = 'auto';

    await generateResponse();
}

/**
 * 调用后端 /api/chat 并处理 AG-UI 标准 SSE 事件
 */
async function generateResponse() {
    isGenerating = true;
    updateUIState();
    // 清除所有旧的建议问题
    document.querySelectorAll('.suggested-questions').forEach(function(el) { el.remove(); });

    const aiMessageEl = appendMessage('assistant', '');
    const textEl = aiMessageEl.querySelector('.message-text');
    var lastToolRef = aiMessageEl.querySelector('.message-role'); // 工具调用插入参考点

    textEl.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';

    abortController = new AbortController();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: currentThreadId ? [messages[messages.length - 1]] : messages,
                model: modelSelect.value,
                thread_id: currentThreadId || undefined,
            }),
            signal: abortController.signal,
        });

        if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        textEl.innerHTML = '';

        // toolCallElements 通过闭包在此函数内共享
        const toolCallElements = {};

        function handleEvent(data) {
            switch (data.type) {
                case 'TEXT_MESSAGE_START':
                    if (textEl._fullContent) {
                        textEl._fullContent += '\n';
                    }
                    break;

                case 'TEXT_MESSAGE_CONTENT':
                    textEl._fullContent = (textEl._fullContent || '') + data.delta;
                    textEl.innerHTML = marked.parse(textEl._fullContent) + '<span class="typing-cursor"></span>';
                    textEl.querySelectorAll('pre code').forEach(function (block) {
                        hljs.highlightElement(block);
                    });
                    scrollToBottom();
                    break;

                case 'TEXT_MESSAGE_END':
                    break;

                case 'TOOL_CALL_START':
                    {
                        var tcContainer = document.createElement('div');
                        tcContainer.className = 'tool-call';
                        tcContainer.innerHTML =
                            '<div class="tool-call-header" data-tool-id="' + escapeHtml(data.toolCallId) + '">' +
                                '<span class="tool-call-icon">🔧</span>' +
                                '<span class="tool-call-name">' + escapeHtml(data.toolCallName) + '</span>' +
                                '<span class="tool-call-status">执行中...</span>' +
                                '<span class="tool-call-arrow">▶</span>' +
                            '</div>' +
                            '<div class="tool-call-body" style="display:none;">' +
                                '<div class="tool-call-section">' +
                                    '<div class="tool-call-section-label">参数</div>' +
                                    '<div class="tool-call-args"></div>' +
                                '</div>' +
                                '<div class="tool-call-section">' +
                                    '<div class="tool-call-section-label">结果</div>' +
                                    '<div class="tool-call-result"></div>' +
                                '</div>' +
                            '</div>';

                        var header = tcContainer.querySelector('.tool-call-header');
                        var body = tcContainer.querySelector('.tool-call-body');
                        header.addEventListener('click', function () {
                            var isOpen = body.style.display !== 'none';
                            body.style.display = isOpen ? 'none' : 'block';
                            header.querySelector('.tool-call-arrow').textContent = isOpen ? '▶' : '▼';
                        });

                        // 将工具调用插入到角色标签之后，按顺序叠加
                        lastToolRef.insertAdjacentElement('afterend', tcContainer);
                        lastToolRef = tcContainer;

                        toolCallElements[data.toolCallId] = {
                            container: tcContainer,
                            argsEl: tcContainer.querySelector('.tool-call-args'),
                            resultEl: tcContainer.querySelector('.tool-call-result'),
                            statusEl: tcContainer.querySelector('.tool-call-status'),
                            header: header,
                            body: body
                        };
                    }
                    break;

                case 'TOOL_CALL_ARGS':
                    if (toolCallElements[data.toolCallId]) {
                        var argsEl = toolCallElements[data.toolCallId].argsEl;
                        argsEl.textContent = (argsEl.textContent || '') + data.delta;
                    }
                    break;

                case 'TOOL_CALL_END':
                    if (toolCallElements[data.toolCallId]) {
                        var el = toolCallElements[data.toolCallId];
                        el.statusEl.textContent = '完成';
                        el.statusEl.classList.add('done');
                        try {
                            var parsed = JSON.parse(el.argsEl.textContent);
                            el.argsEl.innerHTML = '<pre><code>' + escapeHtml(JSON.stringify(parsed, null, 2)) + '</code></pre>';
                        } catch (e) {
                            el.argsEl.textContent = el.argsEl.textContent || '(无参数)';
                        }
                    }
                    break;

                case 'TOOL_CALL_RESULT':
                    if (toolCallElements[data.toolCallId]) {
                        var el2 = toolCallElements[data.toolCallId];
                        el2.resultEl.innerHTML = '<pre><code>' + escapeHtml(data.content) + '</code></pre>';
                        el2.body.style.display = 'block';
                        el2.header.querySelector('.tool-call-arrow').textContent = '▼';
                    }
                    break;

                case 'RUN_STARTED':
                    if (data.threadId) {
                        currentThreadId = data.threadId;
                    }
                    break;

                case 'SESSION_TITLE':
                    if (data.threadId && data.title) {
                        upsertSession(data.threadId, data.title);
                    }
                    break;

                case 'RUN_FINISHED':
                    if (textEl._fullContent) {
                        textEl.innerHTML = marked.parse(textEl._fullContent);
                        textEl.querySelectorAll('pre code').forEach(function (block) {
                            hljs.highlightElement(block);
                        });
                    }
                case 'SUGGESTED_QUESTIONS':
                    if (data.questions && Array.isArray(data.questions)) {
                        var sContainer = document.createElement('div');
                        sContainer.className = 'suggested-questions';
                        data.questions.forEach(function(question) {
                            var chip = document.createElement('button');
                            chip.className = 'suggested-question-chip';
                            chip.textContent = question;
                            chip.addEventListener('click', function() {
                                document.querySelectorAll('.suggested-questions').forEach(function(el) { el.remove(); });
                                messageInput.value = question;
                                sendMessage();
                            });
                            sContainer.appendChild(chip);
                        });
                        var mc = aiMessageEl.querySelector('.message-content');
                        if (mc) {
                            mc.appendChild(sContainer);
                        }
                    }
                    break;


                case 'RUN_ERROR':
                    textEl._fullContent = null;
                    textEl.innerHTML = '<span style="color: var(--danger);">错误：' + escapeHtml(data.message) + '</span>';
                    break;

                case 'OPENAI_MESSAGES_UPSERT':
                    if (data.message) {
                        appendOpenAIMessage(data.message);
                    }
                    break;

                case 'OPENAI_MODEL_REQUEST':
                    handleOpenAIModelRequest(data);
                    break;

                case 'OPENAI_MODEL_CHUNK':
                    handleOpenAIModelChunk(data);
                    break;

                case 'OPENAI_MODEL_RESPONSE':
                    handleOpenAIModelResponse(data);
                    break;
            }
        }

        while (true) {
            var readResult = await reader.read();
            if (readResult.done) break;

            buffer += decoder.decode(readResult.value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop();

            for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                if (line.indexOf('data: ') !== 0) continue;

                try {
                    var data = JSON.parse(line.slice(6));
                    handleEvent(data);
                } catch (e) {
                    console.error('SSE 事件处理失败:', e, line);
                }
            }
        }

        // 最终渲染
        if (textEl._fullContent) {
            textEl.innerHTML = marked.parse(textEl._fullContent);
            textEl.querySelectorAll('pre code').forEach(function (block) {
                hljs.highlightElement(block);
            });
        }

        messages.push({ role: 'assistant', content: textEl._fullContent || '' });

    } catch (error) {
        if (error.name === 'AbortError') {
            if (textEl._fullContent) {
                textEl.innerHTML = marked.parse(textEl._fullContent);
                messages.push({ role: 'assistant', content: textEl._fullContent || '' });
            } else {
                textEl.innerHTML = '<span style="color: var(--text-muted);">已停止生成</span>';
            }
        } else {
            textEl.innerHTML = '<span style="color: var(--danger);">请求失败：' + escapeHtml(error.message) + '</span>';
        }
    } finally {
        isGenerating = false;
        abortController = null;
        updateUIState();
        scrollToBottom();
    }
}

/**
 * 发送 PDF-RAG 消息
 */
async function sendPdfMessage() {
    var content = pdfMessageInput.value.trim();
    if (!content || isPdfGenerating) return;

    var welcome = pdfChatMessages.querySelector('.welcome-screen');
    if (welcome) {
        welcome.remove();
    }

    pdfMessages.push({ role: 'user', content: content });
    appendMessage('user', content, pdfChatMessages, 'PDF 用户', '📄');
    pdfMessageInput.value = '';
    pdfMessageInput.style.height = 'auto';

    isPdfGenerating = true;
    pdfSendBtn.disabled = true;

    var aiMessageEl = appendMessage('assistant', '', pdfChatMessages, 'PDF-RAG');
    var textEl = aiMessageEl.querySelector('.message-text');
    textEl.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';

    try {
        var response = await fetch('/api/pdf-rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: content,
                model: modelSelect.value,
            }),
        });

        if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
        }

        var payload = await response.json();
        var data = payload.data || {};
        var answer = data.answer || '未返回回答';
        var passages = data.passages || [];

        var sourceLines = passages.map(function (item, idx) {
            return '- ' + (idx + 1) + '. ' + (item.source || '未知来源') + ' (score=' + Number(item.score || 0).toFixed(4) + ')';
        });
        var rendered = sourceLines.length > 0
            ? answer + '\n\n---\n**检索来源**\n' + sourceLines.join('\n')
            : answer;
        textEl.innerHTML = marked.parse(rendered);
        pdfMessages.push({ role: 'assistant', content: rendered });
        scrollToBottom(pdfChatMessages);
    } catch (error) {
        textEl.innerHTML = '<span style="color: var(--danger);">PDF-RAG 请求失败：' + escapeHtml(error.message) + '</span>';
    } finally {
        isPdfGenerating = false;
        pdfSendBtn.disabled = false;
    }
}

function stopGeneration() {
    if (abortController) {
        abortController.abort();
    }
}

// ==================== UI 辅助函数 ====================

function appendMessage(role, content, container, roleNameOverride, avatarOverride) {
    if (!container) container = chatMessages;

    var messageEl = document.createElement('div');
    messageEl.className = 'message ' + role;

    var avatar = avatarOverride || (role === 'user' ? 'You' : '🤖');
    var roleName = roleNameOverride || (role === 'user' ? '你' : '百炼 AI');
    var renderedContent = role === 'user' ? escapeHtml(content) : (content ? marked.parse(content) : '');

    messageEl.innerHTML =
        '<div class="message-wrapper">' +
            '<div class="message-avatar">' + avatar + '</div>' +
            '<div class="message-content">' +
                '<div class="message-role">' + roleName + '</div>' +
                '<div class="message-text">' + renderedContent + '</div>' +
            '</div>' +
        '</div>';

    container.appendChild(messageEl);
    scrollToBottom(container);

    return messageEl;
}

function updateUIState() {
    if (isGenerating) {
        sendBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        messageInput.disabled = true;
    } else {
        sendBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
        messageInput.disabled = false;
        messageInput.focus();
    }
}

function scrollToBottom(container) {
    if (!container) container = chatMessages;
    requestAnimationFrame(function () {
        container.scrollTop = container.scrollHeight;
    });
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function createWelcomeScreen() {
    var div = document.createElement('div');
    div.className = 'welcome-screen';
    div.id = 'welcomeScreen';
    div.innerHTML =
        '<div class="welcome-icon">🤖</div>' +
        '<h1>欢迎使用百炼 AI 对话</h1>' +
        '<p>基于阿里云百炼大模型，开始你的智能对话之旅</p>' +
        '<div class="quick-prompts">' +
            '<button class="prompt-btn" data-prompt="请介绍一下你自己">介绍一下你自己</button>' +
            '<button class="prompt-btn" data-prompt="用Python写一个快速排序算法">写一个快排算法</button>' +
            '<button class="prompt-btn" data-prompt="帮我解释一下什么是机器学习">什么是机器学习</button>' +
            '<button class="prompt-btn" data-prompt="给我讲一个有趣的编程笑话">讲个编程笑话</button>' +
        '</div>';

    div.querySelectorAll('.prompt-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            messageInput.value = btn.dataset.prompt;
            sendMessage();
        });
    });

    return div;
}

function createPdfWelcomeScreen() {
    var div = document.createElement('div');
    div.className = 'welcome-screen';
    div.innerHTML =
        '<div class="welcome-icon">📄</div>' +
        '<h1>PDF-RAG 对话</h1>' +
        '<p>输入问题后将基于 kaoqin.pdf 切分后的 Milvus 数据回答</p>';
    return div;
}

function switchTab(tabName) {
    if (tabName === activeTab) return;
    activeTab = tabName;
    chatTabs.forEach(function (btn) {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    var normalActive = tabName === 'normal';
    normalTabPanel.classList.toggle('active', normalActive);
    pdfTabPanel.classList.toggle('active', !normalActive);
    if (normalActive) {
        messageInput.focus();
    } else {
        pdfMessageInput.focus();
    }
}

// ==================== OpenAI 交互面板 ====================

function clearOpenAIInspector() {
    openaiCallIndex = 0;
    Object.keys(openaiCallElements).forEach(function (k) { delete openaiCallElements[k]; });
    openaiMessagesEmpty = true;
    openaiCallsEmpty = true;
    if (openaiMessagesEl) {
        openaiMessagesEl.innerHTML = '<div class="openai-empty">对话开始后将显示 messages 时间线</div>';
    }
    if (openaiModelCallsEl) {
        openaiModelCallsEl.innerHTML = '<div class="openai-empty">模型请求/响应将显示在此处</div>';
    }
}

function clearOpenAIModelCalls() {
    openaiCallIndex = 0;
    Object.keys(openaiCallElements).forEach(function (k) { delete openaiCallElements[k]; });
    openaiCallsEmpty = true;
    if (openaiModelCallsEl) {
        openaiModelCallsEl.innerHTML = '<div class="openai-empty">模型请求/响应将显示在此处</div>';
    }
}

function prettyJson(obj) {
    try {
        return JSON.stringify(obj, null, 2);
    } catch (e) {
        return String(obj);
    }
}

function openAIMessageSummary(msg) {
    if (!msg) return '';
    if (msg.tool_calls && msg.tool_calls.length) {
        var names = msg.tool_calls.map(function (tc) {
            if (tc.function && tc.function.name) return tc.function.name;
            return tc.name || 'tool';
        });
        return 'tool_calls: ' + names.join(', ');
    }
    if (msg.role === 'tool') {
        var preview = (msg.content || '').replace(/\s+/g, ' ').slice(0, 80);
        return 'tool_call_id=' + (msg.tool_call_id || '') + (preview ? ' · ' + preview : '');
    }
    var text = (msg.content == null ? '' : String(msg.content)).replace(/\s+/g, ' ').slice(0, 100);
    return text || '(empty)';
}

function appendOpenAIMessage(msg) {
    if (!openaiMessagesEl || !msg) return;
    if (openaiMessagesEmpty) {
        openaiMessagesEl.innerHTML = '';
        openaiMessagesEmpty = false;
    }

    var card = document.createElement('div');
    var hasTools = !!(msg.tool_calls && msg.tool_calls.length);
    card.className = 'openai-msg-card' + (hasTools ? ' has-tools' : '');

    var role = msg.role || 'unknown';
    card.innerHTML =
        '<div class="openai-msg-header">' +
            '<span class="openai-role-badge ' + escapeHtml(role) + '">' + escapeHtml(role) + '</span>' +
            '<span class="openai-msg-summary">' + escapeHtml(openAIMessageSummary(msg)) + '</span>' +
            '<span class="openai-msg-arrow">▶</span>' +
        '</div>' +
        '<div class="openai-msg-body"><pre><code></code></pre></div>';

    var body = card.querySelector('.openai-msg-body');
    var code = body.querySelector('code');
    code.textContent = prettyJson(msg);

    var header = card.querySelector('.openai-msg-header');
    var arrow = card.querySelector('.openai-msg-arrow');
    header.addEventListener('click', function () {
        var open = body.classList.toggle('open');
        arrow.textContent = open ? '▼' : '▶';
    });

    // 有 tool_calls 或 tool 结果时默认展开
    if (hasTools || role === 'tool') {
        body.classList.add('open');
        arrow.textContent = '▼';
    }

    openaiMessagesEl.appendChild(card);
    openaiMessagesEl.scrollTop = openaiMessagesEl.scrollHeight;
}

function setOpenAIMessagesFromHistory(histMsgs) {
    clearOpenAIInspector();
    if (!histMsgs || !histMsgs.length) return;
    for (var i = 0; i < histMsgs.length; i++) {
        appendOpenAIMessage(histMsgs[i]);
    }
    // 历史回填后清空 Model Calls（仅保留 Messages）
    clearOpenAIModelCalls();
}

function handleOpenAIModelRequest(data) {
    if (!openaiModelCallsEl || !data) return;
    if (openaiCallsEmpty) {
        openaiModelCallsEl.innerHTML = '';
        openaiCallsEmpty = false;
    }

    openaiCallIndex += 1;
    var callId = data.callId || ('call-' + openaiCallIndex);
    var card = document.createElement('div');
    card.className = 'openai-call-card';
    card.dataset.callId = callId;
    card.innerHTML =
        '<div class="openai-call-header">' +
            '<span class="openai-call-index">Call ' + openaiCallIndex + '</span>' +
            '<span class="openai-msg-summary">request → …</span>' +
            '<span class="openai-call-finish"></span>' +
            '<span class="openai-msg-arrow">▼</span>' +
        '</div>' +
        '<div class="openai-call-body open">' +
            '<div class="openai-call-label">Request</div>' +
            '<pre class="openai-call-request"><code></code></pre>' +
            '<div class="openai-call-label">Chunks</div>' +
            '<pre class="openai-call-chunks"><code></code></pre>' +
            '<div class="openai-call-label">Response</div>' +
            '<pre class="openai-call-response"><code>(streaming…)</code></pre>' +
        '</div>';

    var body = card.querySelector('.openai-call-body');
    var arrow = card.querySelector('.openai-msg-arrow');
    card.querySelector('.openai-call-header').addEventListener('click', function () {
        var open = body.classList.toggle('open');
        arrow.textContent = open ? '▼' : '▶';
    });

    card.querySelector('.openai-call-request code').textContent = prettyJson(data.request || {});

    openaiModelCallsEl.appendChild(card);
    openaiModelCallsEl.scrollTop = openaiModelCallsEl.scrollHeight;

    openaiCallElements[callId] = {
        card: card,
        chunks: [],
        chunksEl: card.querySelector('.openai-call-chunks code'),
        responseEl: card.querySelector('.openai-call-response code'),
        finishEl: card.querySelector('.openai-call-finish'),
        summaryEl: card.querySelector('.openai-msg-summary'),
    };
}

function handleOpenAIModelChunk(data) {
    if (!data || !data.callId || !openaiCallElements[data.callId]) return;
    var el = openaiCallElements[data.callId];
    var chunk = data.chunk || null;
    // 兼容旧载荷：choices[0].delta
    if (!chunk && data.choices && data.choices[0]) {
        chunk = {
            object: 'chat.completion.chunk',
            choices: data.choices,
        };
    }
    if (!chunk) return;
    el.chunks.push(chunk);
    el.chunksEl.textContent = prettyJson(el.chunks);
    if (openaiModelCallsEl) {
        openaiModelCallsEl.scrollTop = openaiModelCallsEl.scrollHeight;
    }
}

function handleOpenAIModelResponse(data) {
    if (!data || !data.callId || !openaiCallElements[data.callId]) return;
    var el = openaiCallElements[data.callId];
    var response = data.response || null;
    // 兼容旧载荷
    if (!response && data.message) {
        response = {
            object: 'chat.completion',
            choices: [{
                index: 0,
                message: data.message,
                finish_reason: data.finish_reason || null,
            }],
        };
    }
    if (!response) return;

    el.responseEl.textContent = prettyJson(response);

    var finishReason = null;
    var message = null;
    if (response.choices && response.choices[0]) {
        finishReason = response.choices[0].finish_reason;
        message = response.choices[0].message;
    }
    el.finishEl.textContent = finishReason || '';

    var summary = 'response';
    if (finishReason === 'tool_calls') {
        summary = 'tool_calls';
    } else if (message && message.content) {
        summary = String(message.content).replace(/\s+/g, ' ').slice(0, 60);
    }
    el.summaryEl.textContent = summary;
    if (openaiModelCallsEl) {
        openaiModelCallsEl.scrollTop = openaiModelCallsEl.scrollHeight;
    }
}

// ==================== 会话回放：工具调用渲染辅助函数 ====================

function normalizeToolCallForUI(tc) {
    if (!tc) return { id: '', name: 'unknown', args: {} };
    if (tc.function) {
        var args = {};
        try {
            args = JSON.parse(tc.function.arguments || '{}');
        } catch (e) {
            args = { raw: tc.function.arguments };
        }
        return {
            id: tc.id || '',
            name: tc.function.name || 'unknown',
            args: args,
        };
    }
    return {
        id: tc.id || '',
        name: tc.name || 'unknown',
        args: tc.args || {},
    };
}

function createToolCallElement(tc) {
    tc = normalizeToolCallForUI(tc);
    var tcContainer = document.createElement('div');
    tcContainer.className = 'tool-call';
    var argsJson = '';
    try {
        argsJson = JSON.stringify(tc.args, null, 2);
    } catch (e) {
        argsJson = String(tc.args || '');
    }
    tcContainer.innerHTML =
        '<div class="tool-call-header" data-tool-id="' + escapeHtml(tc.id) + '">' +
            '<span class="tool-call-icon">🔧</span>' +
            '<span class="tool-call-name">' + escapeHtml(tc.name) + '</span>' +
            '<span class="tool-call-status done">完成</span>' +
            '<span class="tool-call-arrow">▶</span>' +
        '</div>' +
        '<div class="tool-call-body" style="display:none;">' +
            '<div class="tool-call-section">' +
                '<div class="tool-call-section-label">参数</div>' +
                '<div class="tool-call-args"><pre><code>' + escapeHtml(argsJson) + '</code></pre></div>' +
            '</div>' +
            '<div class="tool-call-section">' +
                '<div class="tool-call-section-label">结果</div>' +
                '<div class="tool-call-result"></div>' +
            '</div>' +
        '</div>';

    var header = tcContainer.querySelector('.tool-call-header');
    var body = tcContainer.querySelector('.tool-call-body');
    header.addEventListener('click', function () {
        var isOpen = body.style.display !== 'none';
        body.style.display = isOpen ? 'none' : 'block';
        header.querySelector('.tool-call-arrow').textContent = isOpen ? '▶' : '▼';
    });
    return tcContainer;
}

function fillToolCallResult(tcContainer, content) {
    var resultEl = tcContainer.querySelector('.tool-call-result');
    var body = tcContainer.querySelector('.tool-call-body');
    var header = tcContainer.querySelector('.tool-call-header');
    if (resultEl) {
        resultEl.innerHTML = '<pre><code>' + escapeHtml(content) + '</code></pre>';
    }
    body.style.display = 'block';
    header.querySelector('.tool-call-arrow').textContent = '▼';
}

// ==================== 会话管理 ====================
const sessionList = document.getElementById('sessionList');
let sessions = [];

async function loadSessions() {
    try {
        const res = await fetch('/api/sessions');
        const payload = await res.json();
        if (payload.code === 200) {
            sessions = payload.data || [];
            renderSessionList();
        }
    } catch (e) {
        console.error('加载会话列表失败:', e);
    }
}

function renderSessionList() {
    if (!sessionList) return;
    sessionList.innerHTML = '';
    sessions.forEach(function(s) {
        var item = document.createElement('div');
        item.className = 'session-item' + (s.thread_id === currentThreadId ? ' active' : '');
        item.dataset.threadId = s.thread_id;

        var title = document.createElement('span');
        title.className = 'session-item-title';
        title.textContent = s.title || '新对话';
        title.title = s.title || s.thread_id;

        var delBtn = document.createElement('button');
        delBtn.className = 'session-item-delete';
        delBtn.innerHTML = '&times;';
        delBtn.title = '删除会话';
        delBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            deleteSession(s.thread_id);
        });

        item.appendChild(title);
        item.appendChild(delBtn);

        item.addEventListener('click', function() {
            switchSession(s.thread_id);
        });

        sessionList.appendChild(item);
    });
}

async function switchSession(threadId) {
    if (threadId === currentThreadId) return;
    currentThreadId = threadId;
    renderSessionList();

    messages = [];
    chatMessages.innerHTML = '';
    var wEl = document.getElementById('welcomeScreen');
    if (wEl) wEl.remove();
    clearOpenAIInspector();

    try {
        var res = await fetch('/api/sessions/' + threadId + '/messages');
        var payload = await res.json();
        var histMsgs = payload.data || [];
        var toolCallElements = {};

        setOpenAIMessagesFromHistory(histMsgs);

        for (var i = 0; i < histMsgs.length; i++) {
            var msg = histMsgs[i];
            messages.push(msg);

            if (msg.role === 'assistant' && msg.tool_calls && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
                var aiMsgEl = appendMessage('assistant', msg.content || '');
                var lastToolRef = aiMsgEl.querySelector('.message-role');

                for (var j = 0; j < msg.tool_calls.length; j++) {
                    var tc = msg.tool_calls[j];
                    var normalized = normalizeToolCallForUI(tc);
                    var tcContainer = createToolCallElement(normalized);
                    lastToolRef.insertAdjacentElement('afterend', tcContainer);
                    lastToolRef = tcContainer;
                    toolCallElements[normalized.id] = tcContainer;
                }
            } else if (msg.role === 'tool' && msg.tool_call_id && toolCallElements[msg.tool_call_id]) {
                fillToolCallResult(toolCallElements[msg.tool_call_id], msg.content);
            } else if (msg.role === 'tool' || msg.role === 'system') {
                continue;
            } else {
                appendMessage(msg.role, msg.content || '');
            }
        }
        scrollToBottom();
    } catch(e) {
        console.error('加载历史消息失败:', e);
    }
}

async function deleteSession(threadId) {
    if (!confirm('确定要删除这个对话吗？')) return;
    try {
        await fetch('/api/sessions/' + threadId, { method: 'DELETE' });
        sessions = sessions.filter(function(s) { return s.thread_id !== threadId; });
        if (threadId === currentThreadId) {
            currentThreadId = null;
            messages = [];
            chatMessages.innerHTML = '';
            chatMessages.appendChild(createWelcomeScreen());
            clearOpenAIInspector();
        }
        renderSessionList();
    } catch(e) {
        console.error('删除会话失败:', e);
    }
}

loadSessions();

function upsertSession(threadId, title) {
    // 如果 sidebar 还没渲染，先加载列表
    if (!sessions.length) {
        sessions.unshift({
            thread_id: threadId,
            title: title,
            updated_at: new Date().toISOString()
        });
        renderSessionList();
        return;
    }
    // 更新已有或插入新项
    var existing = sessions.find(function(s) { return s.thread_id === threadId; });
    if (existing) {
        existing.title = title;
        renderSessionList();
    } else {
        sessions.unshift({
            thread_id: threadId,
            title: title,
            updated_at: new Date().toISOString()
        });
        renderSessionList();
    }
}
