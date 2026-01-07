window.addEventListener('pageshow', function (event) {
  const uploadBtn = document.getElementById('uploadBtn');
  const uploadStatus = document.getElementById('uploadStatus');
  const receiptImage = document.getElementById('receiptImage');
  const apiKeyDisplay = document.getElementById('api_key');
  const spinner = document.querySelector('.spinner');
  const container = document.querySelector('.container');
  const countdownDisplay = document.getElementById('countdown');

  if (!uploadBtn || !uploadStatus || !receiptImage || !apiKeyDisplay) {
    console.warn("receipt.html: ไม่พบองค์ประกอบที่จำเป็น (uploadBtn, uploadStatus, receiptImage, api_key)");
    return;
  }

  const PAYMENT_PAGE_URL = '../apikey/view-api-keys.html';
  const REDIRECT_DELAY_MS = 2500;

  const STORAGE_KEYS = {
    countdownStart: "countdown_start_time",
    countdownDeadline: "countdown_deadline",
    qrCodeUrl: "qr_code_url",
    orderMeta: "active_order_meta"
  };

  const storage = {
    get(key) {
      return localStorage.getItem(key) ?? sessionStorage.getItem(key);
    },
    set(key, value) {
      localStorage.setItem(key, value);
      sessionStorage.setItem(key, value);
    },
    remove(key) {
      localStorage.removeItem(key);
      sessionStorage.removeItem(key);
    }
  };

  const FIVE_MINUTES = 300; // วินาที
  let countdownInterval = null;
  let isUploading = false;
  let redirectScheduled = false;
  let redirectAfterUpload = false;
  let cancelOrderRequest = null;

  // ใช้แถบเตือนคงที่จาก receipt.html (id="uploadWarning")
  const uploadWarning = document.getElementById('uploadWarning');

  function showUploadWarning() {
    if (uploadWarning) uploadWarning.style.display = 'block';
  }

  function hideUploadWarning() {
    if (uploadWarning) uploadWarning.style.display = 'none';
  }

  function setUploadStatus(message, tone = "info") {
    const colors = {
      success: "green",
      error: "#d9534f",
      info: "#0d6efd"
    };
    uploadStatus.style.color = colors[tone] || colors.info;
    uploadStatus.textContent = message;
  }

  function scheduleRedirect(message) {
    if (redirectScheduled) {
      return;
    }
    redirectScheduled = true;
    if (message) {
      setUploadStatus(message, "info");
    }
    setTimeout(() => {
      window.location.href = PAYMENT_PAGE_URL;
    }, REDIRECT_DELAY_MS);
  }

  function disableUpload(reasonText) {
    uploadBtn.disabled = true;
    uploadBtn.style.backgroundColor = "#ccc";
    uploadBtn.style.cursor = "not-allowed";
    if (reasonText) {
      setUploadStatus(reasonText, "error");
    }
  }

  function enableUpload() {
    uploadBtn.disabled = false;
    uploadBtn.style.backgroundColor = "";
    uploadBtn.style.cursor = "pointer";
  }

  function getCountdownStart() {
    const raw = storage.get(STORAGE_KEYS.countdownStart);
    if (!raw) {
      return null;
    }
    const parsed = parseInt(raw, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function getCountdownDeadline() {
    const raw = storage.get(STORAGE_KEYS.countdownDeadline);
    if (raw) {
      const parsed = parseInt(raw, 10);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
    const start = getCountdownStart();
    if (!start) {
      return null;
    }
    const fallback = start + FIVE_MINUTES * 1000;
    storage.set(STORAGE_KEYS.countdownDeadline, String(fallback));
    return fallback;
  }

  function clearCountdownState() {
    storage.remove(STORAGE_KEYS.countdownStart);
    storage.remove(STORAGE_KEYS.countdownDeadline);
  }

  function clearOrderState() {
    clearCountdownState();
    storage.remove(STORAGE_KEYS.qrCodeUrl);
    storage.remove(STORAGE_KEYS.orderMeta);
  }

  function cancelActiveOrder(reason = "timeout") {
    if (cancelOrderRequest) {
      return cancelOrderRequest;
    }
    const token = localStorage.getItem('token');
    if (!token || !window.API_BASE_URL) {
      return null;
    }
    const orderMeta = getOrderMeta();
    const payload = {
      reason
    };
    if (orderMeta && orderMeta.ref_code) {
      payload.ref_code = orderMeta.ref_code;
    }
    cancelOrderRequest = (async () => {
      try {
        const response = await fetch(`${window.API_BASE_URL}/cancel-order`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.warn(
            "receipt.html: ยกเลิกคำสั่งซื้อไม่สำเร็จ:",
            errorData.detail || errorData.message || response.status
          );
        }
      } catch (err) {
        console.error("receipt.html: เกิดข้อผิดพลาดระหว่างยกเลิกคำสั่งซื้อ", err);
      } finally {
        cancelOrderRequest = null;
      }
    })();
    return cancelOrderRequest;
  }

  function getOrderMeta() {
    const raw = storage.get(STORAGE_KEYS.orderMeta);
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw);
    } catch (err) {
      console.warn("การแปลง order meta ล้มเหลว:", err);
      storage.remove(STORAGE_KEYS.orderMeta);
      return null;
    }
  }

  function hasActiveOrder() {
    return Boolean(getOrderMeta());
  }

  function stopCountdown() {
    if (countdownInterval) {
      clearInterval(countdownInterval);
      countdownInterval = null;
    }
  }

  function handleCountdownExpired() {
    stopCountdown();
    const expiredDuringUpload = isUploading;
    if (!expiredDuringUpload) {
      cancelActiveOrder("countdown-expired");
      clearOrderState();
    }
    if (countdownDisplay) {
      if (expiredDuringUpload) {
        countdownDisplay.textContent = "กำลังตรวจสอบสลิป...";
        countdownDisplay.style.color = "#0d6efd";
      } else {
        countdownDisplay.textContent = "หมดเวลา!";
        countdownDisplay.style.color = "red";
      }
    }
    if (expiredDuringUpload) {
      redirectAfterUpload = true;
      setUploadStatus(
        "หมดเวลาแล้ว แต่ระบบกำลังตรวจสอบสลิปที่ส่งอยู่ให้เสร็จก่อน",
        "info"
      );
    } else {
      setUploadStatus(
        "❌ คุณไม่ได้อัปโหลดสลิปภายในเวลาที่กำหนด ระบบกำลังกลับไปหน้าชำระเงิน",
        "error"
      );
      disableUpload("เวลาหมด ระบบจะกลับไปหน้าสร้าง Apikey");
      scheduleRedirect();
    }
  }

  function showNoTimerMessage() {
    stopCountdown();
    if (countdownDisplay) {
      countdownDisplay.textContent = "ไม่มีเวลาเหลือ";
      countdownDisplay.style.color = "gray";
    }
    if (hasActiveOrder()) {
      cancelActiveOrder("missing-countdown");
    }
    clearOrderState();
    disableUpload("ไม่พบตัวนับเวลา กรุณากลับไปหน้าชำระเงิน");
    scheduleRedirect();
  }

  function updateCountdownDisplay() {
    if (!countdownDisplay) {
      return;
    }
    const deadline = getCountdownDeadline();
    if (!deadline) {
      showNoTimerMessage();
      return;
    }
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      handleCountdownExpired();
      return;
    }
    const remainingSeconds = Math.ceil(remainingMs / 1000);
    const minutes = Math.floor(remainingSeconds / 60);
    const seconds = remainingSeconds % 60;
    countdownDisplay.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    countdownDisplay.style.color = "#d9534f";
  }

  function ensureCountdown() {
    const deadline = getCountdownDeadline();
    if (!deadline) {
      showNoTimerMessage();
      return false;
    }
    if (deadline <= Date.now()) {
      handleCountdownExpired();
      return false;
    }
    stopCountdown();
    updateCountdownDisplay();
    countdownInterval = setInterval(updateCountdownDisplay, 1000);
    return true;
  }

  function ensureOrderReady() {
    if (!hasActiveOrder()) {
      setUploadStatus(
        "❌ ไม่พบคำสั่งซื้อที่รอการตรวจสอบ กรุณากลับไปหน้าชำระเงิน",
        "error"
      );
      disableUpload("กรุณาสร้างคำสั่งซื้อใหม่");
      return false;
    }
    const deadline = getCountdownDeadline();
    if (!deadline) {
      showNoTimerMessage();
      return false;
    }
    if (deadline <= Date.now()) {
      handleCountdownExpired();
      return false;
    }
    return true;
  }

  function handleOrderRemoved() {
    if (isUploading) {
      return;
    }
    stopCountdown();
    if (countdownDisplay) {
      countdownDisplay.textContent = "ไม่มีคำสั่งซื้อ";
      countdownDisplay.style.color = "gray";
    }
    disableUpload("ไม่พบคำสั่งซื้อ กรุณากลับไปหน้าชำระเงิน");
    setUploadStatus("ไม่พบคำสั่งซื้อ กรุณาสร้างคำสั่งซื้อใหม่", "info");
    scheduleRedirect();
  }

  window.addEventListener('storage', (event) => {
    if (!event.key) {
      return;
    }
    if (
      event.key === STORAGE_KEYS.countdownStart ||
      event.key === STORAGE_KEYS.countdownDeadline
    ) {
      ensureCountdown();
    }
    if (event.key === STORAGE_KEYS.orderMeta && !event.newValue) {
      handleOrderRemoved();
    }
  });

  // --- Add: warn user when they try to leave while an upload is in progress ---
  // Show browser's leave-confirm when user attempts to close/refresh/navigate away
  window.addEventListener('beforeunload', function (e) {
    if (isUploading) {
      // Custom text may be ignored by modern browsers, but returnValue triggers the dialog.
      const msg = "การเปลี่ยนหน้าจะยกเลิกการประมวลผลสลิปที่กำลังส่งอยู่ หากออกจากหน้านี้ การประมวลผลจะถูกยกเลิก";
      e.preventDefault();
      e.returnValue = msg;
      return msg;
    }
    // no return -> no prompt
  });

  // Intercept link clicks (delegated) and ask for confirmation if uploading.
  // Allow anchors with href starting with '#' and links that open in a new tab.
  document.addEventListener('click', function (ev) {
    if (!isUploading) return;
    const a = ev.target.closest && ev.target.closest('a');
    if (!a) return;
    const href = a.getAttribute('href');
    const target = a.getAttribute('target');
    if (!href || href.startsWith('#') || target === '_blank') return;
    const leave = confirm("การเปลี่ยนหน้าจะยกเลิกการประมวลผลสลิปที่กำลังส่งอยู่ คุณต้องการออกจากหน้านี้จริงหรือไม่?");
    if (!leave) {
      ev.preventDefault();
      ev.stopPropagation();
    }
  }, true);

  // Intercept form submissions while uploading
  document.addEventListener('submit', function (ev) {
    if (!isUploading) return;
    const leave = confirm("การส่งฟอร์มหรือเปลี่ยนหน้าในขณะนี้จะยกเลิกการประมวลผลสลิปที่กำลังส่งอยู่ คุณต้องการดำเนินการต่อหรือไม่?");
    if (!leave) {
      ev.preventDefault();
      ev.stopPropagation();
    }
  }, true);

  const orderAvailable = hasActiveOrder();
  if (orderAvailable) {
    const timerOk = ensureCountdown();
    if (timerOk) {
      enableUpload();
    }
  } else {
    showNoTimerMessage();
    setUploadStatus(
      "ไม่พบคำสั่งซื้อ กรุณากลับไปหน้าชำระเงินเพื่อขอสั่งซื้อใหม่",
      "error"
    );
  }

  uploadBtn.addEventListener('click', async function () {
    const file = receiptImage.files[0];
    if (!file) {
      setUploadStatus('กรุณาเลือกไฟล์ใบเสร็จ', "error");
      return;
    }

    const token = localStorage.getItem('token');
    if (!token) {
      setUploadStatus('⚠️ กรุณาเข้าสู่ระบบก่อน', "error");
      return;
    }

    if (!ensureOrderReady()) {
      return;
    }

    isUploading = true;
    showUploadWarning(); // แสดงแถบเตือนคงที่ (ภาษาไทย)
    container.classList.add('loading');
    if (spinner) {
      spinner.style.display = 'block';
    }
    setUploadStatus('กำลังประมวลผล...', "info");
    apiKeyDisplay.textContent = '';

    const formData = new FormData();
    formData.append('receipt', file);

    try {
      const response = await fetch(`${window.API_BASE_URL}/upload-receipt`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (response.status === 401) {
        alert("เซสชันของคุณหมดอายุ กรุณาล็อกอินใหม่");
        localStorage.removeItem('token');
        window.location.href = '../login-singup/login.html'; // หรือหน้า login จริงของคุณ
        return;
      }
      const data = await response.json();
      console.log("ข้อมูลตอบกลับจากเซิร์ฟเวอร์:", data);

      if (response.ok && data.success) {
        setUploadStatus('✅ อัปโหลดสำเร็จ!', "success");
        apiKeyDisplay.textContent = data.api_key;
        clearOrderState();
        disableUpload("ได้รับ API Key แล้ว");
      } else {
        const errorMessage = data.detail || data.message || 'ไม่ทราบสาเหตุ';
        setUploadStatus('❌ ' + errorMessage, "error");
      }
    } catch (error) {
      console.error('เกิดข้อผิดพลาด:', error);
      setUploadStatus("❌ ไม่สามารถอัปโหลดสลิปได้ กรุณาลองใหม่อีกครั้ง", "error");
    } finally {
      isUploading = false;
      hideUploadWarning(); // ซ่อนแถบเตือนเมื่อเสร็จ
      container.classList.remove('loading');
      if (spinner) {
        spinner.style.display = '';
      }
      if (redirectAfterUpload) {
        clearOrderState();
        disableUpload("เวลาหมด ระบบจะกลับไปหน้าหลัก");
        scheduleRedirect();
        redirectAfterUpload = false;
      } else if (hasActiveOrder()) {
        ensureCountdown();
      }
    }
  });
});