document.addEventListener('DOMContentLoaded', function () {
  const token = localStorage.getItem("token");
  if (!token) {
    alert("กรุณาล็อกอินก่อน");
    return;
  }

  const amountStored = sessionStorage.getItem("selectedAmount");
  const durationMonths = parseInt(sessionStorage.getItem("selectedDurationMonths") || "1", 10);
  const packageType = sessionStorage.getItem("selectedPackage") || "image";
  const analysisTypes = JSON.parse(sessionStorage.getItem("selectedAnalysis") || "[]");
  const thresholds = JSON.parse(sessionStorage.getItem("selectedThresholds") || "{}");
  const outputModes = JSON.parse(sessionStorage.getItem("selectedOutputModes") || "[]");

  const amountSpan = document.getElementById('amount');
  const qrCodeImage = document.getElementById('qrCodeImage');
  const paymentStatus = document.getElementById('paymentStatus');
  const countdownDisplay = document.getElementById('countdown');
  const confirmPaymentBtn = document.getElementById('confirmPaymentBtn');
  const confirmationHint = document.getElementById('confirmationHint');

  if (!qrCodeImage || !countdownDisplay || !confirmPaymentBtn) {
    console.warn("องค์ประกอบสำคัญหายไป ไม่สามารถทำงานได้");
    return;
  }

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
  let isGeneratingOrder = false;
  let cancelOrderRequest = null;

  const initialAmount = (() => {
    if (amountStored !== null) {
      return amountStored;
    }
    const priceTable = {
      image: 79,
      video: 119,
      both: 159
    };
    return (priceTable[packageType] || 0) * durationMonths;
  })();
  if (amountSpan) {
    amountSpan.innerText = initialAmount;
  }

  function setStatus(message, tone = "info") {
    if (!paymentStatus) {
      return;
    }
    const colors = {
      success: "green",
      error: "#d9534f",
      info: "#0d6efd"
    };
    paymentStatus.style.color = colors[tone] || colors.info;
    paymentStatus.textContent = message;
  }

  function disableConfirm(hintText) {
    confirmPaymentBtn.disabled = true;
    confirmPaymentBtn.style.backgroundColor = "#ccc";
    confirmPaymentBtn.style.cursor = "not-allowed";
    if (confirmationHint && hintText) {
      confirmationHint.textContent = hintText;
    }
  }

  function enableConfirm() {
    confirmPaymentBtn.disabled = false;
    confirmPaymentBtn.style.backgroundColor = "";
    confirmPaymentBtn.style.cursor = "pointer";
    if (confirmationHint) {
      confirmationHint.textContent = "พร้อมไปหน้าอัปโหลดสลิปได้ทันที";
    }
  }

  function setCountdownStart(timestamp) {
    storage.set(STORAGE_KEYS.countdownStart, String(timestamp));
    storage.set(
      STORAGE_KEYS.countdownDeadline,
      String(timestamp + FIVE_MINUTES * 1000)
    );
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

  function persistQrCode(url) {
    storage.set(STORAGE_KEYS.qrCodeUrl, url);
  }

  function getStoredQrCode() {
    return storage.get(STORAGE_KEYS.qrCodeUrl);
  }

  function persistOrderMeta(meta) {
    storage.set(STORAGE_KEYS.orderMeta, JSON.stringify(meta));
  }

  function getOrderMeta() {
    const raw = storage.get(STORAGE_KEYS.orderMeta);
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw);
    } catch (err) {
      console.warn("ไม่สามารถแปลง order meta:", err);
      storage.remove(STORAGE_KEYS.orderMeta);
      return null;
    }
  }

  function cancelActiveOrder(reason = "timeout") {
    if (cancelOrderRequest) {
      return cancelOrderRequest;
    }
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
            "payment.html: ยกเลิกคำสั่งซื้อไม่สำเร็จ:",
            errorData.detail || errorData.message || response.status
          );
        }
      } catch (err) {
        console.error("payment.html: เกิดข้อผิดพลาดระหว่างยกเลิกคำสั่งซื้อ", err);
      } finally {
        cancelOrderRequest = null;
      }
    })();
    return cancelOrderRequest;
  }

  function updateCountdown(deadline) {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      countdownDisplay.textContent = "หมดเวลา!";
      countdownDisplay.style.color = "red";
      stopCountdown();
      onCountdownExpired();
      return;
    }
    const remainingSeconds = Math.ceil(remainingMs / 1000);
    const minutes = Math.floor(remainingSeconds / 60);
    const seconds = remainingSeconds % 60;
    countdownDisplay.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    countdownDisplay.style.color = "#d9534f";
  }

  function stopCountdown() {
    if (countdownInterval) {
      clearInterval(countdownInterval);
      countdownInterval = null;
    }
  }

  function runCountdown(deadline) {
    stopCountdown();
    updateCountdown(deadline);
    countdownInterval = setInterval(() => updateCountdown(deadline), 1000);
  }

  function resumeCountdownIfPossible() {
    const deadline = getCountdownDeadline();
    if (!deadline) {
      return false;
    }
    if (deadline <= Date.now()) {
      onCountdownExpired();
      return false;
    }
    runCountdown(deadline);
    return true;
  }

  function onCountdownExpired() {
    cancelActiveOrder("countdown-expired");
    clearOrderState();
    disableConfirm("คำสั่งซื้อหมดเวลา กำลังสร้างใหม่...");
    setStatus("คำสั่งซื้อหมดเวลา ระบบกำลังสร้าง QR Code ใหม่", "error");
    if (!isGeneratingOrder) {
      requestNewOrder();
    }
  }

  async function requestNewOrder() {
    if (isGeneratingOrder) {
      return;
    }
    isGeneratingOrder = true;
    try {
      setStatus("กำลังสร้าง QR Code ...", "info");
      disableConfirm("กำลังสร้างคำสั่งซื้อใหม่...");
      const response = await fetch(`${window.API_BASE_URL}/generate_qr`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          plan: 'premium',
          package: packageType,
          duration_months: durationMonths,
          analysis_types: analysisTypes,
          thresholds: thresholds,
          output_modes: outputModes
        })
      });

      if (response.status === 401) {
        alert("เซสชันของคุณหมดอายุ กรุณาล็อกอินใหม่");
        localStorage.removeItem('token');
        window.location.href = '../login-singup/login.html'; // หรือหน้า login จริงของคุณ
        return;
      }

      const data = await response.json();
      if (!response.ok || !data.qr_code_url) {
        const errorMessage = data.detail || data.message || 'ไม่สามารถสร้าง QR Code ได้';
        throw new Error(errorMessage);
      }

      persistQrCode(data.qr_code_url);
      qrCodeImage.src = data.qr_code_url;
      if (data.amount !== undefined) {
        sessionStorage.setItem("selectedAmount", data.amount);
        if (amountSpan) {
          amountSpan.innerText = data.amount;
        }
      }
      if (Array.isArray(data.media_access)) {
        sessionStorage.setItem("selectedMediaAccess", JSON.stringify(data.media_access));
      }
      if (typeof data.duration_months === 'number') {
        sessionStorage.setItem("selectedDurationMonths", data.duration_months);
      }

      persistOrderMeta({
        ref_code: data.ref_code,
        amount: data.amount,
        package: data.package || packageType,
        duration_months: data.duration_months || durationMonths,
        created_at: data.created_at || new Date().toISOString()
      });

      const newStart = Date.now();
      setCountdownStart(newStart);
      runCountdown(newStart + FIVE_MINUTES * 1000);
      enableConfirm();
      setStatus(data.message || "พร้อมให้ชำระเงิน", "success");
    } catch (error) {
      console.error("เกิดข้อผิดพลาด:", error);
      setStatus(error.message || "เกิดข้อผิดพลาดในการเชื่อมต่อเซิร์ฟเวอร์", "error");
    } finally {
      isGeneratingOrder = false;
    }
  }

  confirmPaymentBtn.addEventListener('click', (event) => {
    event.preventDefault();
    const orderMeta = getOrderMeta();
    const deadline = getCountdownDeadline();
    if (!orderMeta || !deadline) {
      setStatus("ไม่พบคำสั่งซื้อที่รอชำระ กรุณาสร้าง QR ใหม่", "error");
      cancelActiveOrder("missing-order-data");
      clearOrderState();
      requestNewOrder();
      return;
    }
    if (deadline <= Date.now()) {
      onCountdownExpired();
      return;
    }
    window.location.href = '../receipt/receipt.html';
  });

  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) {
      resumeCountdownIfPossible();
    }
  });

  window.addEventListener('storage', (event) => {
    if (!event.key) {
      return;
    }
    if (
      event.key === STORAGE_KEYS.countdownStart ||
      event.key === STORAGE_KEYS.countdownDeadline
    ) {
      resumeCountdownIfPossible();
    }
    if (event.key === STORAGE_KEYS.orderMeta && !event.newValue) {
      clearCountdownState();
      disableConfirm("ไม่พบคำสั่งซื้อ รอสร้างใหม่...");
      setStatus("คำสั่งซื้อถูกปิดแล้ว โปรดสร้างคำสั่งซื้อใหม่", "info");
    }
  });

  const existingQrUrl = getStoredQrCode();
  if (existingQrUrl) {
    qrCodeImage.src = existingQrUrl;
  }

  const hasActiveCountdown = resumeCountdownIfPossible();
  if (existingQrUrl && hasActiveCountdown && getOrderMeta()) {
    enableConfirm();
    setStatus("พร้อมให้ชำระเงิน", "success");
  } else {
    disableConfirm("กำลังสร้างคำสั่งซื้อ...");
    requestNewOrder();
  }
});