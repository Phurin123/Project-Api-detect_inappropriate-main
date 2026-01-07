function updateThresholdValue(option) {
    const slider = document.getElementById(`${option}-threshold`);
    const display = document.getElementById(`${option}-threshold-value`);
    if (slider && display) {
        display.textContent = parseFloat(slider.value).toFixed(2);
    }
}

function generateApiKey() {
    // ดึง token แทน email
    let token = localStorage.getItem('token');
    if (!token) {
        alert("กรุณาล็อกอินก่อน");
        return;
    }

    let analysisTypes = [];
    let thresholds = {};
    document.querySelectorAll('.analysis-option:checked').forEach(option => {
        analysisTypes.push(option.value);
        let threshold = document.getElementById(option.value + '-threshold').value;
        thresholds[option.value] = parseFloat(threshold);
    });

    const outputModes = Array.from(document.querySelectorAll('.output-option:checked')).map(option => option.value);

    if (analysisTypes.length === 0) {
        alert("กรุณาเลือกโมเดลอย่างน้อย 1 รายการ");
        return;
    }

    fetch(`${window.API_BASE_URL}/request-api-key`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}` // <<-- แนบ token แทน email
        },
        body: JSON.stringify({
            analysis_types: analysisTypes,
            thresholds: thresholds,
            output_modes: outputModes,
            plan: 'test'
        })
    })
        .then(response => {
            // 🔴 เพิ่มการตรวจสอบ 401 (token หมดอายุ) <<-- ใส่ตรงนี้
            if (response.status === 401) {
                alert("เซสชันของคุณหมดอายุ กรุณาล็อกอินใหม่");
                localStorage.removeItem('token');
                window.location.href = '../login-singup/login.html'; // หรือหน้า login จริงของคุณ
                return;
            }
            return response.json(); // ถ้าไม่ใช่ 401 ให้ parse JSON ต่อ
        })
        .then(data => {
            if (!data) return; // ป้องกันกรณี return จาก 401
            if (data.error) {
                alert(data.error);
            } else if (data.apiKey) {
                alert(`API Key Created!\nKey: ${data.apiKey}\nExpires: ${data.expires_at}`);
                window.location.href = 'view-api-keys.html';

            } else {
                alert("คุณเคยขอ API Key free ไปแล้ว");
            }
        })
        .catch(error => {
            console.error("เกิดข้อผิดพลาดในการเชื่อมต่อ:", error);
            alert("เกิดข้อผิดพลาดในการเชื่อมต่อกับ server");
        });
}