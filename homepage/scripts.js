function collectModelSelections() {
  const selectedModels = [];
  const modelThresholds = {};

  document.querySelectorAll('input[name="analysis"]:checked').forEach((checkbox) => {
    const model = checkbox.value;
    selectedModels.push(model);
    const thresholdInput = document.getElementById(`${model}-threshold`);
    const thresholdValue = parseFloat(thresholdInput?.value) || 0.5;
    modelThresholds[model] = thresholdValue;
  });

  if (!selectedModels.length) {
    alert('กรุณาเลือกโมเดลอย่างน้อย 1 โมเดลก่อนอัปโหลดค่ะ');
    return null;
  }

  return {
    selectedModels,
    modelThresholds
  };
}

function collectUniqueLabels(source) {
  const labels = [];
  const seen = new Set();

  const addLabel = (value) => {
    if (typeof value !== 'string') {
      return;
    }
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    labels.push(trimmed);
  };

  const traverse = (item) => {
    if (!item) return;
    if (Array.isArray(item)) {
      item.forEach(traverse);
      return;
    }
    if (typeof item === 'string') {
      addLabel(item);
      return;
    }
    if (typeof item === 'object') {
      if (item.label) {
        addLabel(item.label);
      }
      if (Array.isArray(item.detections)) {
        traverse(item.detections);
      }
    }
  };

  traverse(source);
  return labels;
}

function resetMediaDisplay() {
  const imagePreview = document.getElementById('imagePreview');
  const processedImage = document.getElementById('processedImage');
  const processedVideo = document.getElementById('processedVideo');
  const blurredVideo = document.getElementById('blurredVideo');
  const containers = [
    'imageProcessedContainer',
    'videoProcessedContainer',
    'videoBlurredContainer',
    'imageGalleryContainer',
  ];

  [imagePreview, processedImage].forEach((img) => {
    if (!img) return;
    img.style.display = 'none';
    img.src = '';
  });

  [processedVideo, blurredVideo].forEach((video) => {
    if (!video) return;
    if (video.dataset && video.dataset.objectUrl) {
      URL.revokeObjectURL(video.dataset.objectUrl);
      delete video.dataset.objectUrl;
    }
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.style.display = 'none';
  });

  containers.forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.style.display = 'none';
    }
  });
}

function setLoadingState(isLoading) {
  const loadingSpinner = document.getElementById('loadingSpinner');
  if (!loadingSpinner) return;
  loadingSpinner.style.display = isLoading ? 'block' : 'none';
  if (isLoading) {
    setResultMessage('', 'info');
  }
}

function setResultMessage(message, type = 'info') {
  const resultText = document.getElementById('resultText');
  if (!resultText) {
    return;
  }
  resultText.textContent = message || '';
  const colors = {
    success: '#2ecc71',
    error: '#e74c3c',
    info: '#f1c40f',
  };
  resultText.style.color = colors[type] || '#ecf0f1';
}

// --- New Unified Upload Handler ---

async function handleFileUpload() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*,.zip'; // Allow images and zip
  input.multiple = true; // Allow multiple file selection

  input.onchange = async () => {
    const files = Array.from(input.files || []);
    if (!files.length) return;

    // Always treat as multiple/folder upload to show the Gallery view
    await processMultipleImages(files);
  };

  input.click();
}

async function processSingleImage(file) {
  // This function is no longer used but kept for potential rollback or reference.
  // Logic moved to processMultipleImages to unify the UI output.
  await processMultipleImages([file]);
}

function validateImageUploadCount(files) {
  if (files.length > 100) {
    alert('คุณสามารถอัปโหลดรูปภาพได้สูงสุด 100 รูปเท่านั้น');
    return false;
  }
  return true;
}

// Add validation for video duration
async function validateVideoDuration(file) {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';

    video.onloadedmetadata = () => {
      window.URL.revokeObjectURL(video.src);
      const duration = video.duration;
      if (duration > 60) {
        alert('วิดีโอที่อัปโหลดต้องมีความยาวไม่เกิน 1 นาที');
        resolve(false);
      } else {
        resolve(true);
      }
    };

    video.onerror = () => {
      alert('ไม่สามารถตรวจสอบความยาวของวิดีโอได้');
      resolve(false);
    };

    video.src = URL.createObjectURL(file);
  });
}

async function processMultipleImages(files) {
  if (!validateImageUploadCount(files)) return;

  const selection = collectModelSelections();
  if (!selection) return;

  const {
    selectedModels,
    modelThresholds
  } = selection;

  // Check if user selected a zip file
  const hasZip = files.some(f => f.name.toLowerCase().endsWith('.zip'));

  // Only ask for confirmation if multiple files or zip
  if (files.length > 1 || hasZip) {
    let confirmMsg;
    if (hasZip) {
      confirmMsg = `คุณเลือกไฟล์ ZIP\nระบบจะแตกไฟล์และประมวลผลรูปภาพทั้งหมด\n\nดำเนินการต่อหรือไม่?`;
    } else {
      confirmMsg = `คุณเลือก ${files.length} ไฟล์\nแต่ละไฟล์จะถูกประมวลผลพร้อมกัน\n\nดำเนินการต่อหรือไม่?`;
    }

    if (!confirm(confirmMsg)) return;
  }

  resetMediaDisplay();

  // Prepare gallery but keep it hidden until processing completes
  const galleryContainer = document.getElementById('imageGalleryContainer');
  const gallery = document.getElementById('imageGallery');
  const galleryCount = document.getElementById('galleryCount');

  if (gallery) gallery.innerHTML = '';
  if (galleryContainer) galleryContainer.style.display = 'none';

  setLoadingState(true);
  setResultMessage(`กำลังประมวลผล...`, 'info');

  try {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('images', file); // 'images' field for multiple files
    });
    formData.append('analysis_types', JSON.stringify(selectedModels));
    formData.append('thresholds', JSON.stringify(modelThresholds));

    const response = await fetch(`${window.API_BASE_URL}/analyze-image`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (response.ok) {
      // Handle multi-result response
      const results = Array.isArray(data.results) ? data.results : [data];

      let successCount = 0;
      let failedCount = 0;

      results.forEach(result => {
        const filename = result.original_filename || 'unknown';
        const status = result.status || 'error';
        const detections = Array.isArray(result.detections) ? result.detections : [];
        const labels = collectUniqueLabels(detections);
        const processedImageUrl = result.processed_image_url;

        if (status === 'passed') {
          successCount++;
        } else if (status === 'failed') {
          failedCount++;
        } else {
          failedCount++;
        }

        // Add to gallery
        if (gallery) {
          const galleryItem = document.createElement('div');
          galleryItem.className = 'gallery-item';

          if (processedImageUrl && status !== 'error') {
            const img = document.createElement('img');
            img.src = processedImageUrl;
            img.alt = filename;
            galleryItem.appendChild(img);
          }

          const title = document.createElement('div');
          title.className = 'gallery-item-title';
          title.textContent = filename;
          galleryItem.appendChild(title);

          const statusBadge = document.createElement('div');
          if (status === 'passed') {
            statusBadge.className = 'gallery-item-status passed';
            statusBadge.textContent = '✅ ผ่าน';
          } else if (status === 'failed') {
            statusBadge.className = 'gallery-item-status failed';
            statusBadge.textContent = '❌ ไม่ผ่าน';
          } else {
            statusBadge.className = 'gallery-item-status error';
            statusBadge.textContent = '⚠️ ข้อผิดพลาด';
          }
          galleryItem.appendChild(statusBadge);

          const labelsDiv = document.createElement('div');
          labelsDiv.className = 'gallery-item-labels';
          if (status === 'error') {
            labelsDiv.textContent = result.error || 'ไม่สามารถประมวลผลได้';
          } else {
            labelsDiv.textContent = labels.length ? `พบ: ${labels.join(', ')}` : 'ไม่พบวัตถุที่ตรงตามเงื่อนไข';
          }
          galleryItem.appendChild(labelsDiv);

          gallery.appendChild(galleryItem);
        }
      });

      // Update gallery count and show gallery now that results are ready
      if (galleryCount && gallery) {
        galleryCount.textContent = gallery.children.length;
      }
      if (galleryContainer && gallery && gallery.children.length > 0) {
        galleryContainer.style.display = 'block';
      }

      // Display summary
      const finalStatus = failedCount > 0 ? 'error' : 'success';
      setResultMessage(
        `เสร็จสิ้น! ประมวลผล ${results.length} รูป | ✅ ผ่าน: ${successCount} | ❌ ไม่ผ่าน: ${failedCount}`,
        finalStatus
      );
    } else {
      setResultMessage(`ข้อผิดพลาด: ${data.error || 'เกิดข้อผิดพลาด'}`, 'error');
    }
  } catch (error) {
    setResultMessage(`ข้อผิดพลาด: ${error.message || 'ไม่สามารถประมวลผลได้'}`, 'error');
  } finally {
    setLoadingState(false);
  }
}

// Update uploadVideo to validate video duration
async function uploadVideo() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'video/*';

  input.onchange = async () => {
    const file = input.files?. [0];
    if (!file) return;

    const isValid = await validateVideoDuration(file);
    if (!isValid) return;

    const selection = collectModelSelections();
    if (!selection) {
      return;
    }

    const {
      selectedModels,
      modelThresholds
    } = selection;

    const formData = new FormData();
    formData.append('video', file);
    formData.append('analysis_types', JSON.stringify(selectedModels));
    formData.append('thresholds', JSON.stringify(modelThresholds));

    resetMediaDisplay();
    setLoadingState(true);
    // Show processing message for video uploads (same as image flow)
    setResultMessage('กำลังประมวลผล...', 'info');

    const processedVideo = document.getElementById('processedVideo');
    const blurredVideo = document.getElementById('blurredVideo');

    try {
      const response = await fetch(`${window.API_BASE_URL}/analyze-video`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        const status = (data.status || '').toLowerCase();
        const summaryLabels = Array.isArray(data.summary_labels) ?
          data.summary_labels :
          Object.keys(data.summary || {});
        const labels = collectUniqueLabels(
          summaryLabels.length ? summaryLabels : data.detections,
        );
        const resolvedStatus =
          status || (labels.length ? 'failed' : 'passed');
        const labelsSuffix = labels.length ? ` | รายการตรวจพบ: ${labels.join(', ')}` : '';
        const noLabelsSuffix = ' | ไม่พบวัตถุที่ตรงตามเงื่อนไข';

        if (resolvedStatus === 'failed') {
          setResultMessage(
            `ผลลัพธ์: ไม่ผ่านการทดสอบ${labelsSuffix || noLabelsSuffix}`,
            'error',
          );
        } else {
          const successSuffix = labelsSuffix || noLabelsSuffix;
          setResultMessage(`ผลลัพธ์: ผ่านการทดสอบ${successSuffix}`, 'success');
        }

        if (data.processed_video_url && processedVideo) {
          processedVideo.src = `${data.processed_video_url}?t=${Date.now()}`;
          processedVideo.style.display = 'block';
          processedVideo.load();
          const container = document.getElementById('videoProcessedContainer');
          if (container) {
            container.style.display = 'block';
          }
        }

        if (data.processed_blurred_video_url && blurredVideo) {
          blurredVideo.src = `${data.processed_blurred_video_url}?t=${Date.now()}`;
          blurredVideo.style.display = 'block';
          blurredVideo.load();
          const container = document.getElementById('videoBlurredContainer');
          if (container) {
            container.style.display = 'block';
          }
        }
      } else {
        setResultMessage(`ข้อผิดพลาด: ${data.error || 'เกิดข้อผิดพลาด'}`, 'error');
      }
    } catch (error) {
      setResultMessage('ข้อผิดพลาด: ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์', 'error');
    } finally {
      setLoadingState(false);
    }
  };

  input.click();
}

function downloadManual() {
  const url = `${window.API_BASE_URL}/manual`;
  window.location.href = url;
}

function toggleAdvanced() {
  const advancedSection = document.getElementById('advanced-settings');
  if (!advancedSection) return;
  advancedSection.style.display = advancedSection.style.display === 'none' ? 'block' : 'none';
}

function isValidImageURL(url) {
  try {
    const urlObj = new URL(url);
    // Check if URL has a valid protocol
    if (!['http:', 'https:'].includes(urlObj.protocol)) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

async function uploadImageFromURL() {
  const urlInput = document.getElementById('imageUrl');
  const imageUrl = urlInput?.value?.trim();

  if (!imageUrl) {
    alert('กรุณาใส่ URL รูปภาพ');
    return;
  }

  if (!isValidImageURL(imageUrl)) {
    alert('URL ไม่ถูกต้อง กรุณาใส่ URL ที่ถูกต้อง (ต้องขึ้นต้นด้วย http:// หรือ https://');
    return;
  }

  const selection = collectModelSelections();
  if (!selection) {
    return;
  }

  const {
    selectedModels,
    modelThresholds
  } = selection;

  resetMediaDisplay();
  setLoadingState(true);
  setResultMessage('กำลังดาวน์โหลดรูปภาพจาก URL...', 'info');

  try {
    // Fetch image from URL and convert to blob
    const response = await fetch(imageUrl, {
      mode: 'cors'
    });
    if (!response.ok) {
      throw new Error('ไม่สามารถดาวน์โหลดรูปภาพจาก URL ได้');
    }

    const blob = await response.blob();

    // Check if it's an image
    if (!blob.type.startsWith('image/')) {
      throw new Error('URL ไม่ใช่รูปภาพ');
    }

    // Create a file from blob
    const file = new File([blob], 'image_from_url.jpg', {
      type: blob.type
    });

    const formData = new FormData();
    formData.append('images', file);
    formData.append('analysis_types', JSON.stringify(selectedModels));
    formData.append('thresholds', JSON.stringify(modelThresholds));

    // Display preview
    const imagePreview = document.getElementById('imagePreview');
    const processedImage = document.getElementById('processedImage');

    if (imagePreview) {
      imagePreview.src = imageUrl;
      imagePreview.style.display = 'block';
      const container = document.getElementById('imageOriginalContainer');
      if (container) {
        container.style.display = 'block';
      }
    }

    setResultMessage('กำลังประมวลผลรูปภาพ...', 'info');

    // Send to backend for analysis
    const apiResponse = await fetch(`${window.API_BASE_URL}/analyze-image`, {
      method: 'POST',
      body: formData,
    });

    const data = await apiResponse.json();
    if (apiResponse.ok) {
      const detections = Array.isArray(data.detections) ? data.detections : [];
      const labels = collectUniqueLabels(detections);
      const status =
        (typeof data.status === 'string' && data.status.toLowerCase()) ||
        (labels.length ? 'failed' : 'passed');
      const labelsSuffix = labels.length ? ` | รายการตรวจพบ: ${labels.join(', ')}` : '';
      const noLabelsSuffix = ' | ไม่พบวัตถุที่ตรงตามเงื่อนไข';

      if (status === 'failed') {
        setResultMessage(
          `ผลลัพธ์: ไม่ผ่านการทดสอบ${labelsSuffix || noLabelsSuffix}`,
          'error',
        );
      } else {
        setResultMessage(`ผลลัพธ์: ผ่านการทดสอบ${noLabelsSuffix}`, 'success');
      }

      if (data.processed_image_url && processedImage) {
        processedImage.src = data.processed_image_url;
        processedImage.style.display = 'block';
        const container = document.getElementById('imageProcessedContainer');
        if (container) {
          container.style.display = 'block';
        }
      }

      // Clear the input field after successful upload
      if (urlInput) {
        urlInput.value = '';
      }
    } else {
      setResultMessage(`ข้อผิดพลาด: ${data.error || 'เกิดข้อผิดพลาด'}`, 'error');
    }
  } catch (error) {
    setResultMessage(`ข้อผิดพลาด: ${error.message || 'ไม่สามารถดาวน์โหลดหรือประมวลผลรูปภาพได้'}`, 'error');
  } finally {
    setLoadingState(false);
  }
}

// Keep original function names if some specific flow needs them, but mapped to new logic or just removed if unused.
// Window exports
window.uploadImage = () => handleFileUpload(); // Backward compatibility if needed, though button uses handleFileUpload
window.uploadVideo = uploadVideo;
window.downloadManual = downloadManual;
window.toggleAdvanced = toggleAdvanced;
window.uploadImageFromURL = uploadImageFromURL;
window.uploadImageFolder = () => handleFileUpload(); // Backward compatibility
window.handleFileUpload = handleFileUpload;