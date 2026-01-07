// การแสดง/ซ่อนไอคอนสำหรับรหัสผ่าน
    document.getElementById('password').addEventListener('input', function () {
        var passwordField = document.getElementById('password');
        var eyeIcon = document.getElementById('togglePassword');
        eyeIcon.style.display = passwordField.value ? 'flex' : 'none'; // แสดงไอคอนเมื่อมีการกรอกข้อมูล
    });

// การแสดง/ซ่อนไอคอนสำหรับ Confirm Password
    document.getElementById('confirmPassword').addEventListener('input', function () {
        var confirmPasswordField = document.getElementById('confirmPassword');
        var eyeIcon = document.getElementById('toggleConfirmPassword');
        eyeIcon.style.display = confirmPasswordField.value ? 'flex' : 'none'; // แสดงไอคอนเมื่อมีการกรอกข้อมูล
    });

// การแสดง/ซ่อนรหัสผ่าน
    document.getElementById('togglePassword').addEventListener('click', function () {
        var passwordField = document.getElementById('password');
        var type = passwordField.type === 'password' ? 'text' : 'password';
        passwordField.type = type;
        this.textContent = type === 'password' ? '🔒' : '👁️';
    });

// การแสดง/ซ่อนรหัสผ่านสำหรับ Confirm Password
    document.getElementById('toggleConfirmPassword').addEventListener('click', function () {
        var confirmPasswordField = document.getElementById('confirmPassword');
        var type = confirmPasswordField.type === 'password' ? 'text' : 'password';
        confirmPasswordField.type = type;
        this.textContent = type === 'password' ? '🔒' : '👁️';
    });

// ฟังก์ชันตรวจสอบรหัสผ่าน
    function validatePasswords() {
        var password = document.getElementById('password').value;
        var confirmPassword = document.getElementById('confirmPassword').value;
        var errorText = document.getElementById('passwordError');
        var signUpButton = document.getElementById('submitBtn');

        if (password !== confirmPassword) {
            errorText.style.display = 'block'; // แสดงข้อความแจ้งเตือน
            signUpButton.disabled = true; // ปิดปุ่มสมัคร
        } else {
            errorText.style.display = 'none'; // ซ่อนข้อความแจ้งเตือน
            signUpButton.disabled = false; // เปิดปุ่มสมัคร
        }
}

// ตรวจสอบทุกครั้งที่พิมพ์รหัสผ่าน
document.getElementById('password').addEventListener('input', validatePasswords);
document.getElementById('confirmPassword').addEventListener('input', validatePasswords);

//จัดการ singup
document.getElementById('signupForm').addEventListener('submit', function (event) {
    event.preventDefault(); // ป้องกันไม่ให้ฟอร์มส่งข้อมูลแบบปกติ

    // ดึงข้อมูลจากฟอร์ม
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    // ตรวจสอบว่ารหัสผ่านตรงกันหรือไม่
    if (password !== confirmPassword) {
        document.getElementById('passwordError').style.display = 'block';
        return; // ออกจากฟังก์ชันหากรหัสผ่านไม่ตรงกัน
    } else {
        document.getElementById('passwordError').style.display = 'none';
    }

    // ปิดใช้งานปุ่มส่งข้อมูลขณะกำลังส่งคำขอ
    document.getElementById('submitBtn').disabled = true;

    // เตรียมข้อมูลสำหรับการส่ง
    const data = {
        username: username,
        email: email,
        password: password
    };

    // ส่งข้อมูลด้วย Fetch API
    fetch(`${window.API_BASE_URL}/signup`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => {
        if (!response.ok) {
            // หากเซิร์ฟเวอร์ตอบกลับไม่เป็น OK
            return response.text().then(text => { throw new Error(text); });
        }
        return response.json(); // แปลงข้อมูลเป็น JSON หากการตอบกลับ OK
    })
    .then(data => {
        if (data.message === 'Signup successful') {
            alert('สมัครสมาชิกสำเร็จ!');
            window.location.href = 'login.html'; // เปลี่ยนไปที่หน้าเข้าสู่ระบบ
        } else {
            alert(data.message); // แสดงข้อความผิดพลาดจากเซิร์ฟเวอร์
        }
    })
    .catch(error => {
        alert('เกิดข้อผิดพลาด: ' + error.message); // แสดงข้อความผิดพลาด
    })
    .finally(() => {
        // เปิดใช้งานปุ่มส่งข้อมูลหลังจากคำขอเสร็จสิ้น
        document.getElementById('submitBtn').disabled = false;
    });
});
