// animation.js
let emailInput = document.querySelector(".username");  // เปลี่ยนเป็น emailInput
let userPasswordInput = document.querySelector(".password");  // เปลี่ยนเป็น userPasswordInput
let showPasswordButtonUI = document.querySelector(".password-button");  // เปลี่ยนเป็น showPasswordButtonUI
let face = document.querySelector(".face");

userPasswordInput.addEventListener("focus", (event) => {
  document.querySelectorAll(".hand").forEach((hand) => {
    hand.classList.add("hide");
  });
  document.querySelector(".tongue").classList.remove("breath");
});

userPasswordInput.addEventListener("blur", (event) => {
  document.querySelectorAll(".hand").forEach((hand) => {
    hand.classList.remove("hide");
    hand.classList.remove("peek");
  });
  document.querySelector(".tongue").classList.add("breath");
});

emailInput.addEventListener("focus", (event) => {
  let length = Math.min(emailInput.value.length - 16, 19);
  document.querySelectorAll(".hand").forEach((hand) => {
    hand.classList.remove("hide");
    hand.classList.remove("peek");
  });

  face.style.setProperty("--rotate-head", `${-length}deg`);
});

emailInput.addEventListener("blur", (event) => {
  face.style.setProperty("--rotate-head", "0deg");
});

emailInput.addEventListener(
  "input",
  _.throttle((event) => {
    let length = Math.min(event.target.value.length - 16, 19);
    face.style.setProperty("--rotate-head", `${-length}deg`);
  }, 100)
);

// จัดการการเเสดงผลรหัสผ่าน
showPasswordButtonUI.addEventListener("click", (event) => {
  if (userPasswordInput.type === "password") {
      // เปลี่ยนเป็น text (แสดงรหัส)
      userPasswordInput.type = "text";
      showPasswordButtonUI.innerText = "Show"; // เปลี่ยนปุ่มเป็น "Show"
      
      // เอามือออกจากตา
      document.querySelectorAll(".hand").forEach((hand) => {
          hand.classList.remove("hide"); 
          hand.classList.add("peek");   
      });

  } else {
      // เปลี่ยนกลับเป็น password (ซ่อนรหัส)
      userPasswordInput.type = "password";
      showPasswordButtonUI.innerText = "Hide"; // เปลี่ยนปุ่มเป็น "Hide"
      
      // เอามือกลับมาปิดตา
      document.querySelectorAll(".hand").forEach((hand) => {
          hand.classList.remove("peek");
          hand.classList.add("hide");   
      });
  }
});
