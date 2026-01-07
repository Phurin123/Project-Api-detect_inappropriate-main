document.addEventListener('DOMContentLoaded', () => {
    const loginUrl = '../login-singup/login.html';
    const planButtons = document.querySelectorAll('.plan .btn');
    const hasToken = !!localStorage.getItem('token');

    if (typeof window.refreshMenubarAuthState === 'function') {
        window.refreshMenubarAuthState();
    }

    planButtons.forEach((button) => {
        button.addEventListener('click', (event) => {
            event.preventDefault();
            const targetUrl = button.getAttribute('href');

            if (!hasToken) {
                // ถ้ายังไม่ได้ล็อกอิน → เปิด login.html ในแท็บเดิม
                window.location.href = loginUrl;
            } else {
                // ถ้าล็อกอินแล้ว → เปิดหน้า apikey.html แท็บใหม่
                window.open(targetUrl, '_blank');
            }
        });
    });
});