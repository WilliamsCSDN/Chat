/**
 * 百炼大模型对话 - 前端交互逻辑
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

// ==================== 状态管理 ====================
let messages = [];           // 对话历史
let isGenerating = false;    // 是否正在生成
let abortController = null;  // 用于中止请求

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

// 发送按钮
sendBtn.addEventListener('click', sendMessage);

// 停止按钮
stopBtn.addEventListener('click', stopGeneration);

// 新建对话
newChatBtn.addEventListener('click', () => {
    messages = [];
    chatMessages.innerHTML = '';
    chatMessages.appendChild(createWelcomeScreen());
    messageInput.focus();
});

// 快捷提示按钮
promptBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        messageInput.value = btn.dataset.prompt;
        sendMessage();
    });
});

// 输入框：Enter 发送，Shift+Enter 换行
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating) {
            sendMessage();
        }
    }
});

// 输入框自动调整高度
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
});

// ==================== 核心函数 ====================

/**
 * 发送消息
 */
async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || isGenerating) return;

    // 隐藏欢迎页
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) {
        welcome.remove();
    }

    // 添加用户消息
    messages.push({ role: 'user', content });
    appendMessage('user', content);

    // 清空输入框
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // 开始生成
    await generateResponse();
}

/**
 * 调用后端 API 并流式渲染 AI 回复
 */
async function generateResponse() {
    isGenerating = true;
    updateUIState();

    // 创建 AI 消息容器
    const aiMessageEl = appendMessage('assistant', '');
    const textEl = aiMessageEl.querySelector('.message-text');

    // 显示加载动画
    textEl.innerHTML = `
        <div class="loading-dots">
            <span></span><span></span><span></span>
        </div>
    `;

    let fullContent = '';
    abortController = new AbortController();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: messages,
                model: modelSelect.value,
            }),
            signal: abortController.signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // 清除加载动画
        textEl.innerHTML = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // 解析 SSE 数据
            const lines = buffer.split('\n');
            buffer = lines.pop(); // 保留未完成的行

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.error) {
                            textEl.innerHTML = `<span style="color: var(--danger);">错误：${escapeHtml(data.error)}</span>`;
                            isGenerating = false;
                            updateUIState();
                            return;
                        }

                        if (data.content) {
                            fullContent += data.content;
                            // 使用 Markdown 渲染
                            textEl.innerHTML = marked.parse(fullContent) + '<span class="typing-cursor"></span>';
                            // 代码高亮
                            textEl.querySelectorAll('pre code').forEach(block => {
                                hljs.highlightElement(block);
                            });
                            scrollToBottom();
                        }

                        if (data.done) {
                            // 移除打字光标，最终渲染
                            textEl.innerHTML = marked.parse(fullContent);
                            textEl.querySelectorAll('pre code').forEach(block => {
                                hljs.highlightElement(block);
                            });
                        }
                    } catch (e) {
                        // JSON 解析失败，忽略
                    }
                }
            }
        }

        // 确保最终渲染干净
        if (fullContent) {
            textEl.innerHTML = marked.parse(fullContent);
            textEl.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }

        // 保存 AI 回复到对话历史
        messages.push({ role: 'assistant', content: fullContent });

    } catch (error) {
        if (error.name === 'AbortError') {
            // 用户主动中止
            if (fullContent) {
                textEl.innerHTML = marked.parse(fullContent);
                textEl.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });
                messages.push({ role: 'assistant', content: fullContent });
            } else {
                textEl.innerHTML = '<span style="color: var(--text-muted);">已停止生成</span>';
            }
        } else {
            textEl.innerHTML = `<span style="color: var(--danger);">请求失败：${escapeHtml(error.message)}</span>`;
        }
    } finally {
        isGenerating = false;
        abortController = null;
        updateUIState();
        scrollToBottom();
    }
}

/**
 * 停止生成
 */
function stopGeneration() {
    if (abortController) {
        abortController.abort();
    }
}

// ==================== UI 辅助函数 ====================

/**
 * 添加消息到界面
 */
function appendMessage(role, content) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    const avatar = role === 'user' ? 'You' : '🤖';
    const roleName = role === 'user' ? '你' : '百炼 AI';
    const renderedContent = role === 'user' ? escapeHtml(content) : (content ? marked.parse(content) : '');

    messageEl.innerHTML = `
        <div class="message-wrapper">
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-role">${roleName}</div>
                <div class="message-text">${renderedContent}</div>
            </div>
        </div>
    `;

    chatMessages.appendChild(messageEl);
    scrollToBottom();

    return messageEl;
}

/**
 * 更新 UI 状态（发送/停止按钮切换）
 */
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

/**
 * 滚动到底部
 */
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

/**
 * HTML 转义
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 创建欢迎页面
 */
function createWelcomeScreen() {
    const div = document.createElement('div');
    div.className = 'welcome-screen';
    div.id = 'welcomeScreen';
    div.innerHTML = `
        <div class="welcome-icon">🤖</div>
        <h1>欢迎使用百炼 AI 对话</h1>
        <p>基于阿里云百炼大模型，开始你的智能对话之旅</p>
        <div class="quick-prompts">
            <button class="prompt-btn" data-prompt="请介绍一下你自己">介绍一下你自己</button>
            <button class="prompt-btn" data-prompt="用Python写一个快速排序算法">写一个快排算法</button>
            <button class="prompt-btn" data-prompt="帮我解释一下什么是机器学习">什么是机器学习</button>
            <button class="prompt-btn" data-prompt="给我讲一个有趣的编程笑话">讲个编程笑话</button>
        </div>
    `;

    // 重新绑定快捷提示按钮事件
    div.querySelectorAll('.prompt-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            messageInput.value = btn.dataset.prompt;
            sendMessage();
        });
    });

    return div;
}

