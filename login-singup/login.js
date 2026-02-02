// login.js
let loginButton = document.querySelector(".login-button");
let usernameInput = document.getElementById("username"); // รับข้อมูลจาก input ของ email
let passwordInput = document.getElementById("password"); // รับข้อมูลจาก input ของ password

// จัดการ login
loginButton.addEventListener("click", async (event) => {
  // ป้องกันการโหลดหน้าใหม่
  event.preventDefault();

  // รับค่าจาก input
  let username = usernameInput.value;
  let password = passwordInput.value;

  // ตรวจสอบว่ามีการกรอกข้อมูลหรือไม่
  if (!username || !password) {
    alert("กรุณากรอกชื่อผู้ใช้และรหัสผ่าน");
    return;
  }

  // ส่งข้อมูลไปยัง back-end (API login)
  try {
    let response = await fetch(`${window.API_BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        email: username,
        password: password
      })
    });

    let data = await response.json();

    if (response.ok) {
      // เมื่อเข้าสู่ระบบสำเร็จ
      // เก็บอีเมลใน sessionStorage
      // เก็บ token ลง localStorage
      localStorage.setItem('auth_token', data.token);

      // หลังจาก login สำเร็จ ให้ไปที่หน้า apikey.html
      window.location.href = '../apikey/view-api-keys.html'; // เปลี่ยนเส้นทางไปยังหน้า apikey.html
    } else {
      // แสดงข้อความผิดพลาดจาก back-end
      alert(data.error || "เกิดข้อผิดพลาดในการเข้าสู่ระบบ");
    }
  } catch (error) {
    console.error("เกิดข้อผิดพลาดในการเชื่อมต่อกับ server:", error);
    alert("เกิดข้อผิดพลาดในการเชื่อมต่อกับ server");
  }
});

function loginWithGoogle() {
  // เปลี่ยนเส้นทางไปยัง Google OAuth URL
  window.location.href = `${window.API_BASE_URL}/auth/google`;
}