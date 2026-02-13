(function syncTokenFromUrl() {

    const params = new URLSearchParams(window.location.search);

    const token = params.get('token');

    if (token) {

        localStorage.setItem('token', token);

        const newUrl = window.location.origin + window.location.pathname;

        window.history.replaceState({}, document.title, newUrl);

    }

})();

let hasShownSessionExpiredAlert = false;

function handleUnauthorizedResponse(response) {
    if (response.status !== 401) {
        return false;
    }

    if (hasShownSessionExpiredAlert) {
        return true;
    }

    if (!hasShownSessionExpiredAlert) {
        alert('เซสชันของคุณหมดอายุ กรุณาล็อกอินใหม่');
        hasShownSessionExpiredAlert = true;
    }

    localStorage.removeItem('token');
    window.location.href = '../login-singup/login.html';
    return true;
}

const HISTORY_TOGGLE_SHOW_LABEL = 'View History';
const HISTORY_TOGGLE_HIDE_LABEL = 'Hide History';
const API_DETAILS_SHOW_LABEL = 'View Details';
const API_DETAILS_HIDE_LABEL = 'Hide Details';



function escapeHtml(value) {

    if (value === null || value === undefined) {

        return '';

    }

    return String(value)

        .replace(/&/g, '&amp;')

        .replace(/</g, '&lt;')

        .replace(/>/g, '&gt;')

        .replace(/"/g, '&quot;')

        .replace(/'/g, '&#39;');

}



function parseDate(value) {

    if (!value) {

        return null;

    }

    if (value instanceof Date && !Number.isNaN(value.getTime())) {

        return value;

    }

    if (typeof value === 'string') {

        let candidate = value.trim();
        if (!candidate) {
            return null;
        }

        if (!candidate.includes('T') && candidate.includes(' ')) {

            candidate = candidate.replace(' ', 'T');

        }

        const hasTimezone =
            /([zZ]|[+-]\d{2}:?\d{2})$/.test(candidate);

        let parsed;
        if (hasTimezone) {
            parsed = new Date(candidate);
        } else {
            const isoLike = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?$/.test(candidate);
            parsed = new Date(isoLike ? `${candidate}Z` : candidate);
            if (Number.isNaN(parsed.getTime()) && !isoLike) {
                parsed = new Date(`${candidate}Z`);
            }
        }

        if (parsed && !Number.isNaN(parsed.getTime())) {

            return parsed;

        }

    }

    return null;

}



function formatDateTime(value) {
    const parsed = parseDate(value);
    if (!parsed) {
        return value || '—';
    }

    try {
        // แปลงเวลา UTC ให้เป็นเวลาไทย
        return parsed.toLocaleString('th-TH', {
            dateStyle: 'medium',
            timeStyle: 'short',
            timeZone: 'Asia/Bangkok'
        });
    } catch (err) {
        return parsed.toISOString();
    }
}



function formatQuota(quota) {

    if (quota === -1) {

        return 'ไม่จำกัดการใช้งาน';

    }

    if (quota === null || quota === undefined) {

        return '—';

    }

    return quota;

}



function formatAnalysisTypes(types) {

    if (!Array.isArray(types) || types.length === 0) {

        return '—';

    }

    return types.join(', ');

}



function formatThresholds(thresholds) {

    if (!thresholds || typeof thresholds !== 'object' || Array.isArray(thresholds)) {

        return '—';

    }

    const entries = Object.entries(thresholds);

    if (!entries.length) {

        return '—';

    }

    return entries

        .map(([key, value]) => {

            const numeric = Number.parseFloat(value);

            if (Number.isFinite(numeric)) {

                return `${key}: ${numeric.toFixed(2)}`;

            }

            return `${key}: ${value}`;

        })

        .join(', ');

}



function formatMediaAccess(access) {

    if (!Array.isArray(access) || access.length === 0) {

        return '—';

    }

    const labels = {
        image: 'Image',
        video: 'Video',
    };

    return access

        .map((item) => labels[item] || item)

        .join(', ');

}



function formatOutputModes(modes) {

    if (!Array.isArray(modes) || modes.length === 0) {

        return '—';

    }

    const labels = {
        blur: 'Blur',
        bbox: 'Bounding Box',
    };

    return modes

        .map((mode) => labels[mode] || mode)

        .join(', ');

}



function formatStatusBadge(status) {
    const normalized = (status || '').toLowerCase();

    const labels = {
        passed: 'ผ่าน',
        failed: 'ไม่ผ่าน',
        error: 'ข้อผิดพลาด',
    };

    const safeClass = normalized.replace(/[^a-z0-9-]/g, '') || 'unknown';
    const label = labels[normalized] || status || 'ไม่ทราบสถานะ';
    return `<span class="status-badge status-${safeClass}">${escapeHtml(label)}</span>`;

}





async function fetchUsername() {

    const token = localStorage.getItem('token');

    const usernameDisplay = document.getElementById('usernameDisplay');

    if (!token) {

        usernameDisplay.textContent = '⚠️ กรุณาเข้าสู่ระบบ';

        return;

    }
    try {
        const res = await fetch(`${window.API_BASE_URL}/get-username`, {
            headers: {
                Authorization: `Bearer ${token}`,
            },
        });
        if (handleUnauthorizedResponse(res)) {
            return;
        }
        const data = await res.json();
        if (res.ok && data.username) {
            usernameDisplay.textContent = `👤 สวัสดีคุณ: ${data.username}`;
        } else if (data.error) {
            usernameDisplay.textContent = `👤 ${data.error}`;
        } else {
            usernameDisplay.textContent = '👤 ไม่พบชื่อผู้ใช้';
        }
    } catch (error) {
        console.error('Error fetching username:', error);
        usernameDisplay.textContent = '👤 ดึงชื่อผู้ใช้ไม่สำเร็จ';
    }

}



async function fetchApiKeys() {

    const token = localStorage.getItem('token');

    if (!token) {

        throw new Error('⚠️ กรุณาเข้าสู่ระบบก่อน');

    }



    const response = await fetch(`${window.API_BASE_URL}/get-api-keys`, {

        headers: {

            Authorization: `Bearer ${token}`,

        },

    });

    if (handleUnauthorizedResponse(response)) {
        throw new Error('unauthorized');
    }

    const data = await response.json();



    if (!response.ok || data.error) {

        throw new Error(data.error || 'คุณยังไม่ได้สร้างAPI Keys');

    }



    if (!Array.isArray(data.api_keys)) {

        return [];

    }



    return data.api_keys;

}



async function fetchApiKeyHistory(limit = 50) {

    const token = localStorage.getItem('token');

    if (!token) {

        throw new Error('⚠️ กรุณาเข้าสู่ระบบก่อน');

    }



    const response = await fetch(`${window.API_BASE_URL}/get-api-key-history?limit=${limit}`, {

        headers: {

            Authorization: `Bearer ${token}`,

        },

    });

    if (handleUnauthorizedResponse(response)) {
        throw new Error('unauthorized');
    }

    const data = await response.json();



    if (!response.ok || data.error) {

        throw new Error(data.error || 'เกิดข้อผิดพลาดในการดึงประวัติการใช้งาน');

    }



    if (!Array.isArray(data.history)) {

        return [];

    }



    return data.history;

}

function renderApiKeysWithHistory(apiKeys, historyEntries) {

    const listElement = document.getElementById('apiKeysList');

    const groupedHistory = groupHistoryByKey(Array.isArray(historyEntries) ? historyEntries : []);

    const cards = [];
    let historyIndex = 0;
    let detailsIndex = 0;

    if (Array.isArray(apiKeys) && apiKeys.length) {
        apiKeys.forEach((key) => {
            const lookupKey = key && Object.prototype.hasOwnProperty.call(key, 'api_key') ?
                key.api_key :
                undefined;
            const mapKey = lookupKey ?? null;
            const entries = groupedHistory.get(mapKey) || [];
            groupedHistory.delete(mapKey);
            const historyId = `history-${historyIndex++}`;
            const detailsId = `api-details-${detailsIndex++}`;
            cards.push(createApiKeyCard(key, entries, historyId, detailsId));
        });
    }

    groupedHistory.forEach((entries, orphanKey) => {
        if (!entries.length) {
            return;
        }
        cards.push(createOrphanHistoryCard(orphanKey, entries, `history-${historyIndex++}`));
    });

    if (!cards.length) {
        listElement.innerHTML = '<p>ยังไม่มี API Key สำหรับบัญชีนี้</p>';
        return;
    }

    listElement.innerHTML = cards.join('');

}

function createApiKeyCard(key, historyEntries, historyId, detailsId) {

    const apiKeyText = escapeHtml(key.api_key || '—');

    const analysisText = escapeHtml(formatAnalysisTypes(key.analysis_types));

    const thresholdsText = escapeHtml(formatThresholds(key.thresholds));

    const createdText = escapeHtml(formatDateTime(key.created_at));

    const lastUsedText = escapeHtml(formatDateTime(key.last_used_at));

    const expiresText = escapeHtml(formatDateTime(key.expires_at));

    const planText = escapeHtml(key.plan || '—');

    const packageText = escapeHtml(key.package || '—');

    const usageCount = escapeHtml(typeof key.usage_count === 'number' ? key.usage_count : 0);

    const mediaAccessText = escapeHtml(formatMediaAccess(key.media_access));

    const outputModesText = escapeHtml(formatOutputModes(key.output_modes));

    const historyContent = Array.isArray(historyEntries) && historyEntries.length ?
        historyEntries.map(renderHistoryEntry).join('') :
        `<p class="history-empty">No history available for this API Key</p>`;

    return `

        <div class="api-key">

            <div class="api-key__summary">
                <div class="api-key__summary-info">
                    <p class="api-key__summary-field">
                        <strong>API Key:</strong> <span class="api-key-text">${apiKeyText}</span>
                        <button class="copy-btn" data-key="${apiKeyText}" title="Copy API Key">
                            📋
                        </button>
                    </p>
                    <p class="api-key__summary-field"><strong>Plan:</strong> ${planText}</p>
                    <p class="api-key__summary-field"><strong>Created At:</strong> ${createdText}</p>
                </div>
                <button class="api-key-toggle" type="button" aria-expanded="false" data-target="${detailsId}">
                    <span class="api-key-toggle-label">${API_DETAILS_SHOW_LABEL}</span>
                    <span class="api-key-toggle-icon" aria-hidden="true">▾</span>
                </button>
            </div>

            <div class="api-key__details" id="${detailsId}" aria-hidden="true">
                <div class="api-key__details-content">
                    <p><strong>Media Access:</strong> ${mediaAccessText}</p>
                    <p><strong>Output Modes:</strong> ${outputModesText}</p>
                    <p><strong>Usage Count:</strong> ${usageCount}</p>
                    <p><strong>Analysis Types:</strong> ${analysisText}</p>
                    <p><strong>Thresholds:</strong> ${thresholdsText}</p>
                    <p><strong>Last Used:</strong> ${lastUsedText}</p>
                    ${key.expires_at ? `<p><strong>Expires At:</strong> ${expiresText}</p>` : ''}
                </div>
                ${createHistorySection(historyId, historyContent)}
            </div>

        </div>

    `;

}

function createOrphanHistoryCard(orphanKey, historyEntries, historyId) {

    const safeKey = orphanKey ? escapeHtml(orphanKey) : 'Unknown';

    const historyContent = Array.isArray(historyEntries) && historyEntries.length ?
        historyEntries.map(renderHistoryEntry).join('') :
        `<p class="history-empty">No history available for this API Key</p>`;

    return `

        <div class="api-key api-key--orphan">

            <p><strong>API Key:</strong> ${safeKey}</p>

            <p class="orphan-note">Could not match history entries to current API Key</p>

            ${createHistorySection(historyId, historyContent)}

        </div>

    `;

}

function createHistorySection(historyId, historyContent) {

    return `

        <div class="history-section">

            <button class="history-toggle" type="button" aria-expanded="false" data-target="${historyId}">
                <span class="history-toggle-label">${HISTORY_TOGGLE_SHOW_LABEL}</span>
                <span class="history-toggle-icon" aria-hidden="true">▾</span>
            </button>

            <div class="history-list" id="${historyId}" aria-hidden="true">
                    <p class="history-warning" style="color: #ffca28; font-size: 0.9em; margin: 10px 0; font-style: italic;">
                    ⚠️ Note: Images and videos will be removed from the database within 7 day
                </p>
                ${historyContent}
            </div>

        </div>

    `;

}

function renderHistoryEntry(entry) {

    const statusBadge = formatStatusBadge(entry.status);

    const fileName = escapeHtml(entry.original_filename || '—');
    const createdText = escapeHtml(formatDateTime(entry.created_at));

    const models = escapeHtml(formatAnalysisTypes(entry.analysis_types));

    const thresholds = escapeHtml(formatThresholds(entry.thresholds));

    const mediaAccess = escapeHtml(formatMediaAccess(entry.media_access));

    const outputModes = escapeHtml(formatOutputModes(entry.output_modes));


    const mediaType = (entry.media_type || '').toLowerCase();

    const mediaTypeLabel = mediaType === 'video' ? 'Video' : mediaType === 'image' ? 'Image' : 'Unknown';

    const isVideo = mediaType === 'video';

    const detectionSummary = Array.isArray(entry.detection_summary) && entry.detection_summary.length ?
        escapeHtml(entry.detection_summary.join(', ')) :
        'ไม่มีการตรวจจับ';



    const links = [];

    const ONE_DAY_MS = 24 * 60 * 60 * 1000;
    const now = new Date();
    const createdDate = parseDate(entry.created_at);
    const isExpired = createdDate && (now - createdDate > ONE_DAY_MS);

    if (isExpired) {
        links.push(`<span class="expired-label" style="color: #aaa;">File removed (older than 1 day)</span>`);
    } else if (isVideo) {

        if (entry.processed_video_url) {
            links.push(`<a href='${escapeHtml(entry.processed_video_url)}' target='_blank' rel='noopener'>View Video</a>`);
        }
        if (entry.processed_blurred_video_url) {
            links.push(`<a href='${escapeHtml(entry.processed_blurred_video_url)}' target='_blank' rel='noopener'>View Video (Blurred)</a>`);
        }

    } else {

        if (entry.processed_image_url) {
            links.push(`<a href='${escapeHtml(entry.processed_image_url)}' target='_blank' rel='noopener'>View Image</a>`);
        }
        if (entry.processed_blurred_image_url) {
            links.push(`<a href='${escapeHtml(entry.processed_blurred_image_url)}' target='_blank' rel='noopener'>View Image (Blurred)</a>`);
        }

    }



    const actions = links.length ? `<div class='history-actions'>${links.join('')}</div>` : '';



    return `

        <div class='history-entry'>

            <p><strong>File Name:</strong> ${fileName}</p>
            <p><strong>Status:</strong> ${statusBadge}</p>
            <p><strong>Media Type:</strong> ${escapeHtml(mediaTypeLabel)}</p>
            <p><strong>Detection Summary:</strong> ${detectionSummary}</p>
            <p><strong>Models:</strong> ${models}</p>
            <p><strong>Thresholds:</strong> ${thresholds}</p>
            <p><strong>Media Access:</strong> ${mediaAccess}</p>
            <p><strong>Output Modes:</strong> ${outputModes}</p>
            <p><strong>Created At:</strong> ${createdText}</p>

            ${actions}

        </div>

    `;

}

function groupHistoryByKey(historyEntries) {

    const grouped = new Map();

    if (!Array.isArray(historyEntries)) {
        return grouped;
    }

    historyEntries.forEach((entry) => {
        if (!entry) {
            return;
        }
        const keyValue = Object.prototype.hasOwnProperty.call(entry, 'api_key') ? entry.api_key : undefined;
        const mapKey = keyValue ?? null;

        if (!grouped.has(mapKey)) {
            grouped.set(mapKey, []);
        }

        grouped.get(mapKey).push(entry);
    });

    grouped.forEach((list) => {
        list.sort((a, b) => {
            const dateA = parseDate(a?.created_at);
            const dateB = parseDate(b?.created_at);
            const timeA = dateA ? dateA.getTime() : 0;
            const timeB = dateB ? dateB.getTime() : 0;
            return timeB - timeA;
        });
    });

    return grouped;

}

async function loadApiKeysWithHistory() {

    const listElement = document.getElementById('apiKeysList');

    if (!listElement) {
        return;
    }

    listElement.innerHTML = '<p>กำลังโหลดข้อมูล...</p>';

    try {
        const [apiKeys, historyEntries] = await Promise.all([
            fetchApiKeys(),
            fetchApiKeyHistory(),
        ]);

        renderApiKeysWithHistory(apiKeys, historyEntries);
    } catch (error) {
        if (error && error.message === 'unauthorized') {
            return;
        }
        console.error('Error loading API key data:', error);
        const fallbackMessage = error && error.message ?
            error.message :
            'คุณยังไม่ได้สร้าง API Keys';
        listElement.innerHTML = `<p>${escapeHtml(fallbackMessage)}</p>`;
    }

}

document.addEventListener('click', function (event) {
    const apiKeyToggle = event.target.closest('.api-key-toggle');
    if (apiKeyToggle) {
        const targetId = apiKeyToggle.getAttribute('data-target');
        if (!targetId) {
            return;
        }

        const detailSection = document.getElementById(targetId);
        if (!detailSection) {
            return;
        }

        const isOpen = !detailSection.classList.contains('open');
        detailSection.classList.toggle('open', isOpen);
        detailSection.setAttribute('aria-hidden', isOpen ? 'false' : 'true');

        const label = apiKeyToggle.querySelector('.api-key-toggle-label');
        if (label) {
            label.textContent = isOpen ? API_DETAILS_HIDE_LABEL : API_DETAILS_SHOW_LABEL;
        }

        const icon = apiKeyToggle.querySelector('.api-key-toggle-icon');
        if (icon) {
            icon.textContent = isOpen ? '▴' : '▾';
        }

        apiKeyToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        return;
    }

    const toggleButton = event.target.closest('.history-toggle');
    if (toggleButton) {
        const targetId = toggleButton.getAttribute('data-target');
        if (!targetId) {
            return;
        }

        const historySection = document.getElementById(targetId);
        if (!historySection) {
            return;
        }

        const isOpen = !historySection.classList.contains('open');
        historySection.classList.toggle('open', isOpen);
        if (isOpen) {
            historySection.style.maxHeight = `${historySection.scrollHeight}px`;
            historySection.setAttribute('aria-hidden', 'false');
        } else {
            historySection.style.maxHeight = '0px';
            historySection.setAttribute('aria-hidden', 'true');
        }
        toggleButton.setAttribute('aria-expanded', isOpen ? 'true' : 'false');

        const label = toggleButton.querySelector('.history-toggle-label');
        if (label) {
            label.textContent = isOpen ? HISTORY_TOGGLE_HIDE_LABEL : HISTORY_TOGGLE_SHOW_LABEL;
        }
        const icon = toggleButton.querySelector('.history-toggle-icon');
        if (icon) {
            icon.textContent = isOpen ? '▴' : '▾';
        }
        return;
    }

    const button = event.target.closest('.show-video-btn');
    if (!button) {
        return;
    }

    const container = button.closest('.history-preview');
    if (!container) {
        return;
    }

    const videoUrl = button.getAttribute('data-video-url');
    if (!videoUrl) {
        return;
    }

    const videoElement = document.createElement('video');
    videoElement.controls = true;
    videoElement.preload = 'metadata';
    videoElement.src = videoUrl;
    videoElement.setAttribute('playsinline', '');
    videoElement.className = 'history-preview-video';

    container.innerHTML = '';
    container.appendChild(videoElement);
});

document.addEventListener('click', async function (event) {
    const copyBtn = event.target.closest('.copy-btn');
    if (!copyBtn) return;

    const key = copyBtn.getAttribute('data-key');
    if (!key) return;

    try {
        await navigator.clipboard.writeText(key);
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '✅';
        copyBtn.disabled = true;

        setTimeout(() => {
            copyBtn.innerHTML = originalText;
            copyBtn.disabled = false;
        }, 2000);
    } catch (err) {
        console.error('Failed to copy details: ', err);
        alert('ไม่สามารถคัดลอกได้');
    }
});




window.onload = async function () {

    const token = localStorage.getItem('token');
    const apiKeysListElement = document.getElementById('apiKeysList');

    if (!token) {

        document.getElementById('usernameDisplay').textContent = '⚠️ กรุณาเข้าสู่ระบบ';

        if (apiKeysListElement) {
            apiKeysListElement.innerHTML = '<p>⚠️ กรุณาเข้าสู่ระบบก่อน</p>';
        }

        return;

    }

    if (typeof window.refreshMenubarAuthState === 'function') {
        window.refreshMenubarAuthState();
    }



    await fetchUsername();

    await loadApiKeysWithHistory();

};



function logout() {

    localStorage.removeItem('token');
    if (typeof window.refreshMenubarAuthState === 'function') {
        window.refreshMenubarAuthState();
    }

    window.location.href = '../homepage/index.html';

}