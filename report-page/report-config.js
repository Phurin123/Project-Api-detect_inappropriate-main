 // ฟังก์ชันแสดงฟอร์มรายงานปัญหาตามประเภทที่เลือก
 function submitReport() {
    const description = document.getElementById("problemDescription").value;
    const category = document.getElementById("problemSelect").value;
  
    if (description && category) {
      console.log({
        issue: description,
        category: category
      }); // ล็อกข้อมูลที่ส่งไปเพื่อการตรวจสอบ

      fetch(`${window.API_BASE_URL}/report-issue`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          issue: description,
          category: category
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          alert(data.success ? 'ขอบคุณสำหรับการรายงานปัญหาของคุณ!' : 'ไม่สามารถส่งข้อมูลได้ กรุณาลองใหม่อีกครั้ง');
        })
        .catch(() => {
          alert('ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์');
        });
  
      // การแจ้งเตือนเมื่อส่งสำเร็จ
      alert("รายงานปัญหาของคุณถูกส่งเรียบร้อยแล้ว");
      
      // รีเซ็ตฟอร์มหลังจากส่ง
      document.getElementById("problemDescription").value = ''; // Clear the text area
      document.getElementById("problemSelect").value = ''; // Reset the dropdown
    } else {
      alert("กรุณากรอกรายละเอียดปัญหาก่อนส่ง");
    }
  }
